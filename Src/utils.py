from lungmask import mask
import SimpleITK as sitk
import matplotlib.pyplot as plt
#import pydicom
import os
import pdb
import numpy as np
import cv2 as cv
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import math
from scipy.io import savemat
import shutil
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure
import copy  
from scipy import ndimage 
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
#from sklearn.externals import joblib
import pdb
import random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pickle
import statistics
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import scikitplot as skplt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import joblib


IMG_PX_SIZE = 512
MAX = 65535

def make_dirs(path):
    """
    Creates the directory as specified from the path
    in case it exists it deletes it
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)

def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    if (len(r) != 0):
        for i in map_to_bins:
            r[i-1] += 1
            
    return [r, map_to_bins]

def HistHyper(img):
    c = 0.02
    el = math.log(1+(1/c))
    xx, yy = img.shape
    outV = np.zeros((xx,yy))
    a,b = histc(img.flatten(), np.array([i for i in range(1,np.amax(img)+1)]))
    a = a/np.linalg.norm(a)
    cuma = np.cumsum(a, dtype=float)
    normcm = cuma / np.linalg.norm(cuma)
    
    for i in range(0,xx):
        for j in range(0,yy):
            if(img[i,j] == 0):
                ex = math.exp( el * normcm[img[i,j]] -1)
                outV[i,j] = np.uint8(MAX*c*ex)
            else:
                ex = math.exp( el * normcm[img[i,j]-1] -1)
                outV[i,j] = np.uint8(MAX*c*ex)

    return outV



def find_final_severity_score(prediction):
    prediction = prediction
    if (prediction >= 0 and prediction <= 5):
        severity = 0
    elif (prediction > 5 and prediction <= 9):
        severity = 1
    elif (prediction > 9 and prediction <= 13):
        severity = 2
    elif (prediction > 13 and prediction <= 20):
        severity = 3
    else:
        severity = 0
    return severity



def find_score_from_infection(infection_percentage):
    if (infection_percentage >= 0 and infection_percentage <= 25):
        score = 1
    elif (infection_percentage > 25 and infection_percentage <= 50):
        score = 2
    elif (infection_percentage > 50 and infection_percentage <= 75):
        score = 3
    elif (infection_percentage > 75 and infection_percentage <= 100):
        score = 4
    else:
        score = 0
    return score 
    
def find_weighted_score(curr_mask, infection_mask):
    left_lung = copy.deepcopy(curr_mask)
    right_lung = copy.deepcopy(curr_mask)

    right_lung[right_lung == 2] = 0 
    left_lung[left_lung == 1] = 0 
    left_lung[left_lung == 2] = 1
    LW = 2 
    RW = 3 
    filterSize =(2, 2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
    result_new = cv.dilate(infection_mask, kernel)

    if (np.sum(left_lung) == 0):
        infection_left_region = 0
    else: 
        infection_left_region = (np.sum((result_new * left_lung)) / np.sum(left_lung)) * 100

    if (np.sum(right_lung)== 0):
        infection_right_region = 0
    else:
        infection_right_region = (np.sum((result_new * right_lung)) / (np.sum(right_lung))) * 100
    
    if math.isnan(infection_left_region) or math.isnan(infection_right_region):
        final_score = 0;
    else:

        left_score = find_score_from_infection(infection_left_region)
        right_score = find_score_from_infection(infection_right_region)
        final_score = LW * left_score + RW * right_score
    features = [infection_right_region, infection_left_region]
    return final_score, features

def find_connected_component(mask):
    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(mask)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
    sizes = sizes[1:]
    nb_blobs -= 1

    # minimum size of particles we want to keep (number of pixels).
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    min_size = 70  

    # output image with only the kept components
    im_result = np.zeros_like(im_with_separated_blobs)
    # for every component in the image, keep it only if it's above min_size

    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 1

    return im_result

def find_template(curr_mask, Masks, index):
    curr_sum = ((np.sum(curr_mask[curr_mask==2])/2) + np.sum(curr_mask[curr_mask==1]))
    max_sum = curr_sum
    for x in range(index-10, index+10):
        if (x > 0 and x < len(Masks)):
            temp_mask = Masks[x,:,:]
            temp_sum = ((np.sum(temp_mask[temp_mask==2])/2) + np.sum(temp_mask[temp_mask==1]))
            if(temp_sum > max_sum):
                max_sum = temp_sum
                template = Masks[x,:,:]
            else:
                template = curr_mask
            

    return template


def convert_to_HU(image):
    [rows, cols] = image.shape
    new_img = np.zeros([rows,cols])
    for row in range(rows):
        for col in range(cols):
            new_img[row][col] = ((image[row][col] * 1300)/255) - 1024

    return new_img

def window_image(img, window_center,window_width, intercept, slope, rescale=True):
   
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = (img - img_min) / (img_max - img_min)*255.0 
    return img

def Create_input_features(features_list):
   
    len_features = []
    train_features = []
    Training_features = []
    Index_array = []
    min = 500
    max = 0
    for features in features_list:
        temp_features = []
        sum_list = []
        for x in features:
            temp_features.extend(x)
            sum_list.append(sum(x))
        train_features.append(temp_features)
        vals = np.array(sum_list)
        sort_index = np.flip(np.argsort(vals))
        Index_array.append(sort_index)
        len_features.append(len(features))
    
    for i, features in enumerate(train_features):

        if (len(features) < 80):
           
            temp_features = []
            temp_features = features
            summation = [0, 0]
            for j, x in enumerate(features):
                summation[j % 2] += x
            if (len(features) != 0):
                right_value = (summation[0] *2) /len(features)
                left_value = (summation[1] *2) /len(features)
            else:
                right_value = 0
                left_value = 0
            while(len(features)< 80):
                temp_features.append(right_value)
                temp_features.append(left_value)
        else:
           
            index_incr = (np.floor(len(features)/40)).astype("uint8")
            if (index_incr % 2 == 1):
                index_incr = (index_incr - 1).astype("uint16")

            
            randlist = (np.arange(0, len(features),index_incr)).astype("uint16")
            randlist = randlist[0:40]
            
            temp_features = []
            for index in randlist:#[:-1]:
                
                temp_features.append(statistics.median(features[index:(index+index_incr):2]))
                temp_features.append(statistics.median(features[(index+1):(index+index_incr):2]))
           
        Training_features.append(temp_features)
    array_of_features = np.vstack(Training_features)
    return array_of_features

def Evaluate_model(classifier_name, classifier, X_test, y_test, save_dir):

    print('===================================================')
    print('*************'+ classifier_name+'*****************')
    print('===================================================')
    
    if classifier_name == 'FFNN':
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, np.argmax(y_pred, axis = 1))
        print(cm)
        
        ax = plt.subplot()
        sns.set(font_scale=3.0)
        sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
        title_font = {'size':'30'}  # Adjust to fit
        ax.set_title('Confusion Matrix', fontdict=title_font);
        ax.tick_params(axis='both', which='major', labelsize=30)  # Adjust to fit
        ax.xaxis.set_ticklabels(['Mi' , 'Mo', 'Se', 'Cr']);
        ax.yaxis.set_ticklabels(['Mi' , 'Mo', 'Se', 'Cr']);
       
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.show()
        plt.savefig(save_dir+ classifier_name+'_confusionmatrix.png')
        plt.close()
        print(f1_score(y_test, y_pred, average='macro'))
        print(classification_report(y_test, y_pred,target_names=['Mild', 'Moderate', 'Severe', 'Critical']))
        
    else:
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        ax = plt.subplot()
        sns.set(font_scale=3.0)
        sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
        title_font = {'size':'30'}  # Adjust to fit
        ax.set_title('Confusion Matrix', fontdict=title_font);
        ax.tick_params(axis='both', which='major', labelsize=30)  # Adjust to fit
        ax.xaxis.set_ticklabels(['Mi' , 'Mo', 'Se', 'Cr']);
        ax.yaxis.set_ticklabels(['Mi' , 'Mo', 'Se', 'Cr']);
       
         
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.show()
        plt.savefig(save_dir+ classifier_name+'_confusionmatrix.png')
        plt.close()
        print(f1_score(y_test, y_pred, average='macro'))
        print(classification_report(y_test, y_pred,target_names=['Mild', 'Moderate', 'Severe', 'Critical']))

       
def train_classsifier(classifier_name, X_train, y_train, model_dir):
    # Fitting a classifier to the Training set
    if (classifier_name == 'Random_forest'):
        classifier = RandomForestClassifier(n_estimators = 100, random_state=42)
        classifier.fit(X_train, y_train)

    if (classifier_name == 'Logistic_regression'):
        classifier = LogisticRegression(max_iter=1000 )
        classifier.fit(X_train, y_train)
    
    if (classifier_name == 'kNN'):
        classifier = KNeighborsClassifier()
        classifier.fit(X_train, y_train)
    
    if (classifier_name == 'Naive_bayes'):
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
    
    if (classifier_name == 'Decision_tree'):
        classifier = DecisionTreeClassifier(max_depth = 9, random_state = 42)
        classifier.fit(X_train, y_train)

    if (classifier_name == 'SVM'):
        #classifier = svm.SVC(kernel='poly', degree=4, C=1, decision_function_shape='ovo')
        classifier = svm.SVC(kernel='rbf', gamma='scale', C=1, decision_function_shape='ovo',probability=True)
        classifier.fit(X_train, y_train)
    
    if (classifier_name == 'XG_boost'):
        classifier = XGBClassifier(random_state=0)
        classifier.fit(X_train, y_train)
    
    if (classifier_name == 'ExtraTreesClassifier'):
        classifier = ExtraTreesClassifier(random_state=0)
        classifier.fit(X_train, y_train)

    if (classifier_name == 'AdaBoostClassifier'):
        classifier = AdaBoostClassifier(n_estimators=50, random_state=0)
        classifier.fit(X_train, y_train)

    if (classifier_name == 'GradientBoostingClassifier'):
        classifier = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50, max_depth=3), n_estimators=50, learning_rate=1)
        classifier.fit(X_train, y_train)

    if (classifier_name == 'CatBoost'):
        classifier = CatBoostClassifier(logging_level='Silent')
        classifier.fit(X_train, y_train)

    if (classifier_name == 'Neural Network'):
        classifier = MLPClassifier(max_iter = 1000, random_state=42, learning_rate='adaptive', validation_fraction=0.2,
            solver = 'sgd', hidden_layer_sizes = [80, 32])
        classifier.fit(X_train, y_train)

    if (classifier_name == 'FFNN'):
        x_train = X_train.astype('float32') / 100
        num_labels = len(np.unique(y_train))
        y_train = to_categorical(y_train)

        # network parameters
        input_size = 80
        batch_size = 20
        hidden_units = 64
        hidden_units3 = 16
        dropout = 0.3

        classifier = Sequential()
        classifier.add(Dense(hidden_units, input_dim=input_size))
        classifier.add(Activation('relu'))
        classifier.add(Dropout(dropout))

        classifier.add(Dense(hidden_units3))
        classifier.add(Activation('relu'))
        classifier.add(Dropout(dropout))
        classifier.add(Dense(num_labels))
        classifier.add(Activation('softmax'))

        classifier.summary()

        classifier.compile(loss='categorical_crossentropy', 
                      optimizer='adam',
                      metrics=['accuracy'])
        
        classifier.fit(X_train, y_train, epochs=100, batch_size=batch_size)
    
    pickle.dump(classifier, open(model_dir+ classifier_name + '.pkl', 'wb'))
    joblib.dump(classifier, model_dir+ classifier_name + '.sav')

    return classifier


def Generate_patient_severity(predictions, test_patients, classifier_name, result_dir):
    if not os.path.exists(result_dir+classifier_name):
        os.makedirs(result_dir+classifier_name)


    for i, ypred in enumerate(predictions):
        if (ypred == 0):
            with open((result_dir+ classifier_name +"/mild.csv"), "a") as mild:
                mild.write(test_patients[i]+ "\n")
        if (ypred == 1):
            with open((result_dir+ classifier_name +"/moderate.csv"), "a") as moderate:
                moderate.write(test_patients[i]+ "\n")
        if (ypred == 2):
            with open((result_dir+ classifier_name +"/severe.csv"), "a") as severe:
                severe.write(test_patients[i]+ "\n")
        if (ypred == 3):
            with open((result_dir+ classifier_name +"/critical.csv"), "a") as critical:
                critical.write(test_patients[i]+ "\n")