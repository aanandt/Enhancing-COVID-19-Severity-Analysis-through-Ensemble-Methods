import os
import sys
import argparse
import shutil
from tqdm import tqdm
import pandas as pd
from utils import *

from Generate_Lung_Mask import *
from Extract_feature_vector import *



path = '../Dataset/Test/'# Path to the dataset
mask_image_path = '../Temp_dir/Test/Masks_and_Images/' #Path to lung mask and original ct images
temp_dir = '../Temp_dir/Test/All_files/'#Path to store the images, mask, and segmented images
preprocessed_path = '../Temp_dir/Test/Preprocessed_Images/' # Path to the infection segementation

feat_filename = 'features_test.npy' # Test feature filename
feat_dir = '../Features/Test/' # Path to Validation feature directory
result_dir = '../Results/Submissions/' #Path to the results for submission
model_dir = '../Models/SubmissionModels/' # Path to the pickle file for the trained models

shutil.rmtree(result_dir, ignore_errors=True)
if not os.path.exists(feat_dir):
	os.makedirs(feat_dir)
if not os.path.exists(result_dir):
	os.makedirs(result_dir)


Percentage_patientwise_infection = {}
Patientwise_infection_features = []
patients = os.listdir(path)

sorted_patients = sorted(patients, key=lambda x: int(x.split('.')[0].split('_')[3]))
lung_threshold = 0.07
pdb.set_trace()
'''
Pretrained UNET model takes huge time to generate the lung Mask. 
Instead you can use the Mask generated by the authors using the link provided. 
If new dataset is using then  uncomment the following two lines of code to generate the lung mask.
'''

# for patient in tqdm(sorted_patients, desc = 'Generating lung mask for patient\'s CT_scans'):
# 	perform_maskUnetandResizeslices(path, patient, mask_image_path, temp_dir)

''' 
Feature extraction is used with the help of multiprocessing the machine. 
The features extracted for the given challenge dataset also provided.
If a new dataset is using then uncomment the following six lines of code.
'''

num_cores = multiprocessing.cpu_count() - 5
infection_rate = Parallel(n_jobs=num_cores)(delayed(Extract_features)(mask_image_path, patient, preprocessed_path, lung_threshold)for patient in (sorted_patients))
for i, pair_value in enumerate(infection_rate):
	Patientwise_infection_features.append(pair_value[1])
np.save(feat_dir +feat_filename, np.array(Patientwise_infection_features, dtype=object), allow_pickle=True)
X_test = Create_input_features(Patientwise_infection_features)

'''
If the stored feature is using for the training of the model then uncomment the following two lines of code.
Download the features to ../Features/Test/features_train.npy
'''

# features_list = (np.load(os.path.join(feat_dir +feat_filename), allow_pickle=True))
# X_test = Create_input_features(features_list)

# ### Generating predictions for the test data

RF_SVM_ERT_model = pickle.load(open(model_dir + 'Ensemble.pkl', 'rb'))
y_pred_RF_SVM_ERT = RF_SVM_ERT_model.predict(X_test)
Generate_patient_severity(y_pred_RF_SVM_ERT, sorted_patients, 'RF_SVM_ERT_model', result_dir)


XGB_SVM_ERT_model = pickle.load(open(model_dir + 'Ensemble1.pkl', 'rb'))
y_pred_XGB_SVM_ERT = XGB_SVM_ERT_model.predict(X_test)
Generate_patient_severity(y_pred_XGB_SVM_ERT, sorted_patients, 'XGB_SVM_ERT_model', result_dir)


GB_ADB_ERT_model = pickle.load(open(model_dir + 'Ensemble2.pkl', 'rb'))
y_pred_GB_ADB_ERT = GB_ADB_ERT_model.predict(X_test)
Generate_patient_severity(y_pred_GB_ADB_ERT, sorted_patients, 'GB_ADB_ERT_model', result_dir)


XGB_SVM_ERT_TV_model = pickle.load(open(model_dir + 'Ensemble3_tv.pkl', 'rb'))
y_pred_XGB_SVM_ERT_TV = XGB_SVM_ERT_TV_model.predict(X_test)
Generate_patient_severity(y_pred_XGB_SVM_ERT_TV, sorted_patients, 'XGB_SVM_ERT_TV_model', result_dir)


GB_ADB_ERT_TV_model = pickle.load(open(model_dir + 'Ensemble4_tv.pkl', 'rb'))
y_pred_GB_ADB_ERT_TV = GB_ADB_ERT_TV_model.predict(X_test)
Generate_patient_severity(y_pred_GB_ADB_ERT_TV, sorted_patients, 'GB_ADB_ERT_TV_model', result_dir)

