import os
from tqdm import tqdm
import pandas as pd
from utils import *
from Generate_Lung_Mask import *
from Extract_feature_vector import *
import joblib
import pickle


gt_filename = '../Dataset/groundtruth_train.xlsx' # Path to ground truth file in .xlsx format

path = '../Dataset/Train/' #Path to train-dataset
mask_image_path = '../Temp_dir/Train/Masks_and_Images/' #Path to lung mask and original ct images
temp_dir = '../Temp_dir/Train/All_files/' #Path to store the images, mask, and segmented images
preprocessed_path = '../Temp_dir/Train/Preprocessed_Images/' # Path to the infection segementation

feat_filename = 'features_train.npy' # Train feature filename
feat_dir = '../Features/Train/' # Path to Validation feature directory
model_dir = '../Models/saved_models1/'  # Path to the pickle file for the trained models

if not os.path.exists(feat_dir):
	os.makedirs(feat_dir)

if not os.path.exists(model_dir):
	os.makedirs(model_dir)

df = pd.read_excel(gt_filename)
val_Y= list(map(int, df['Category']-1))
y_train = np.asarray(val_Y, dtype=np.int64)
train_patients = list(df['Name'])


Percentage_patientwise_infection = {}
Patientwise_infection_features = []
patients = os.listdir(path)
sorted_patients = sorted(train_patients, key=lambda x: int(x.split('.')[0].split('_')[2]))
lung_threshold = 0.07

'''
Pretrained UNET model takes huge time to generate the lung Mask. 
The authors generated the lung mask and provided a link to download the lung mask for the training data.
Download the lung mask to '../Temp_dir/Train/Masks_and_Images/'
If new dataset is using then  uncomment the following three lines of code to generate the lung mask.
'''

# for patient in tqdm(sorted_patients, desc = 'Generating lung mask for patient\'s CT_scans'):
# 	if patient in train_patients:
# 		masks, images = perform_maskUnetandResizeslices(path, patient, mask_image_path, temp_dir)
		
''' 
Feature extraction is used with the help of multiprocessing the machine. 
The features extracted for the given challenge dataset also provided.
If a new dataset is using then uncomment the following six lines of code.
'''

# num_cores = multiprocessing.cpu_count() - 2
# infection_rate = Parallel(n_jobs=num_cores)(delayed(Extract_features)(path, patient, preprocessed_path, lung_threshold)for patient in (sorted_patients))
# for i, pair_value in enumerate(infection_rate):
# 	Patientwise_infection_features.append(pair_value[1])
#X_train = Create_input_features(Patientwise_infection_features)
# np.save(feat_dir +feat_filename, np.array(Patientwise_infection_features, dtype=object), allow_pickle=True)

'''
If the stored feature is using for the training of the model then uncomment the following two lines of code.
Download the features to ../Features/Train/features_train.npy
'''
features_list = (np.load(os.path.join(feat_dir +feat_filename), allow_pickle=True))

X_train = Create_input_features(features_list)

#### Training the models

lgr_model = train_classsifier('Logistic_regression', X_train, y_train, model_dir)
knn_model = train_classsifier('kNN', X_train, y_train, model_dir)
gnb_model = train_classsifier('Naive_bayes', X_train, y_train, model_dir)
svm_model = train_classsifier('SVM', X_train, y_train, model_dir)
dtree_model = train_classsifier('Decision_tree', X_train, y_train, model_dir)
rf_model = train_classsifier('Random_forest', X_train, y_train, model_dir)
xgb_model = train_classsifier('XG_boost', X_train, y_train, model_dir)
ada_boost_model = train_classsifier('AdaBoostClassifier', X_train, y_train, model_dir)
gradient_boost_model = train_classsifier('GradientBoostingClassifier', X_train, y_train, model_dir)
#cat_boost_model = train_classsifier('CatBoost', X_train, y_train)
extra_tree_model = train_classsifier('ExtraTreesClassifier', X_train, y_train, model_dir)
mlp_model = train_classsifier('Neural Network', X_train, y_train, model_dir)

pdb.set_trace()

estimators=[('rf', rf_model), ('svm', svm_model), ('extra_tree', extra_tree_model)]
ensemble = VotingClassifier(estimators, voting='hard', flatten_transform=True)
ensemble.fit(X_train, y_train)
pickle.dump(ensemble, open(model_dir+ 'Ensemble' + '.pkl', 'wb'))
joblib.dump(ensemble, model_dir+ 'Ensemble' + '.sav')

estimators=[('xgboost_classifier', xgb_model), ('svm', svm_model), ('extra_tree', extra_tree_model)]#, ('decision_tree', dtree_model)]#('cat_boost', cat_boost_model)]#, ('decision_tree', dtree_model)]
ensemble1 = VotingClassifier(estimators, voting='hard',  flatten_transform=True)
ensemble1.fit(X_train, y_train)
pickle.dump(ensemble1, open(model_dir+ 'Ensemble1' + '.pkl', 'wb'))
joblib.dump(ensemble1, model_dir+ 'Ensemble1' + '.sav')

estimators=[('gradient_boost_classifier', gradient_boost_model), ('ada_boost', ada_boost_model), ('extra_tree', extra_tree_model)]#, ('decision_tree', dtree_model)]#('cat_boost', cat_boost_model)]#, ('decision_tree', dtree_model)]
ensemble2 = VotingClassifier(estimators, voting='hard',  flatten_transform=True)
ensemble2.fit(X_train, y_train)
pickle.dump(ensemble2, open(model_dir+ 'Ensemble2' + '.pkl', 'wb'))
joblib.dump(ensemble2, model_dir+ 'Ensemble2' + '.sav')