import os
from tqdm import tqdm
import pandas as pd
from utils import *
import joblib
from Generate_Lung_Mask import *
from Extract_feature_vector import *
import joblib
import pickle


gt_filename = '/speech/tmp/anand/ICASSP_2023/groundtruth_train.xlsx'
gt_filename_val = '/speech/tmp/anand/ICASSP_2023/ground_truth.xlsx'
path = '/cbr/anand/ResearchWork/ICASSP_23/COVID_19_Challenge/SubmissionFolder/TempDir/Train/Masks_and_Images'#'../Temp_dir/Train/Masks_and_Images/'
mask_image_path = '/cbr/anand/ResearchWork/ICASSP_23/COVID_19_Challenge/SubmissionFolder/TempDir/Train/Masks_and_Images'#'../Temp_dir/Train/Masks_and_Images/'
temp_dir = '/cbr/anand/ResearchWork/ICASSP_23/COVID_19_Challenge/SubmissionFolder/TempDir/Train/All_files'#'../Temp_dir/Train/All_files/'
preprocessed_path = '../Temp_dir/Train/Preprocessed_Images/'
feat_filename = 'features_train.npy'
feat_dir = '../Features/Train/'
result_dir = '../Results/'
model_dir = '../Models/saved_models_train_val/'

if not os.path.exists(feat_dir):
	os.makedirs(feat_dir)
if not os.path.exists(result_dir):
	os.makedirs(result_dir)
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

df = pd.read_excel(gt_filename)
val_Y= list(map(int, df['Category']-1))
y_train = np.asarray(val_Y, dtype=np.int64)
train_patients = list(df['Name'])

df = pd.read_excel(gt_filename_val)
val_Y= list(map(int, df['Category']-1))
y_val = np.asarray(val_Y, dtype=np.int64)
val_patients = list(df['Name'])


pdb.set_trace()
Percentage_patientwise_infection = {}
Patientwise_infection_features = []
patients = os.listdir(path)
sorted_patients = sorted(train_patients, key=lambda x: int(x.split('.')[0].split('_')[2]))
lung_threshold = 0.07
# for patient in tqdm(sorted_patients, desc = 'Training patient\'s CT_scans'):
# 	if patient in train_patients:
# 		masks, images = perform_maskUnetandResizeslices(path, patient, mask_image_path, temp_dir)
		
# pdb.set_trace()

# num_cores = multiprocessing.cpu_count() - 5
# infection_rate = Parallel(n_jobs=num_cores)(delayed(Extract_features)(path, patient, preprocessed_path, lung_threshold)for patient in (sorted_patients))
# #severity_score, features = Extract_features(masks, images, patient, preprocessed_path, lung_threshold)
# for i, pair_value in enumerate(infection_rate):
# 	Patientwise_infection_features.append(pair_value[1])
# pdb.set_trace()

# np.save(feat_dir +feat_filename, np.array(Patientwise_infection_features, dtype=object), allow_pickle=True)
features_list = (np.load(os.path.join(feat_dir +feat_filename), allow_pickle=True))
X_train = Create_input_features(features_list)
features_list = (np.load(os.path.join('../Features/Validation/features_Validation.npy'), allow_pickle=True))
X_val = Create_input_features(features_list)

X_train_val = np.concatenate((X_train, X_val), axis=0)
y_train_val = np.concatenate((y_train, y_val))
pdb.set_trace()
#X_train = Create_input_features(Patientwise_infection_features)
#### Training the models

lgr_model = train_classsifier('Logistic_regression', X_train_val, y_train_val, model_dir)
knn_model = train_classsifier('kNN', X_train_val, y_train_val, model_dir)
gnb_model = train_classsifier('Naive_bayes', X_train_val, y_train_val, model_dir)
svm_model = train_classsifier('SVM', X_train_val, y_train_val, model_dir)
dtree_model = train_classsifier('Decision_tree', X_train_val, y_train_val, model_dir)
rf_model = train_classsifier('Random_forest', X_train_val, y_train_val, model_dir)
xgb_model = train_classsifier('XG_boost', X_train_val, y_train_val, model_dir)
ada_boost_model = train_classsifier('AdaBoostClassifier', X_train_val, y_train_val, model_dir)
gradient_boost_model = train_classsifier('GradientBoostingClassifier', X_train_val, y_train_val, model_dir)
#cat_boost_model = train_classsifier('CatBoost', X_train, y_train)
extra_tree_model = train_classsifier('ExtraTreesClassifier', X_train_val, y_train_val, model_dir)
mlp_model = train_classsifier('Neural Network', X_train_val, y_train_val, model_dir)

pdb.set_trace()

# estimators=[('rf', rf_model), ('decision_tree', dtree_model), ('xgboost_classifier', xgb_model)]
# #create our voting classifier, inputting our models
# ensemble = VotingClassifier(estimators, voting='hard')
# ensemble.fit(X_train, y_train)
# pickle.dump(ensemble, open(model_dir+ 'Ensemble' + '.pkl', 'wb'))
# joblib.dump(ensemble, model_dir+ 'Ensemble' + '.sav')

estimators=[('rf', rf_model), ('svm', svm_model), ('extra_tree', extra_tree_model)]
#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard', flatten_transform=True)
ensemble.fit(X_train_val, y_train_val)
pickle.dump(ensemble, open(model_dir+ 'Ensemble' + '.pkl', 'wb'))
joblib.dump(ensemble, model_dir+ 'Ensemble' + '.sav')

estimators=[('xgboost_classifier', xgb_model), ('svm', svm_model), ('extra_tree', extra_tree_model)]#, ('decision_tree', dtree_model)]#('cat_boost', cat_boost_model)]#, ('decision_tree', dtree_model)]
#create our voting classifier, inputting our models
ensemble1 = VotingClassifier(estimators, voting='hard',  flatten_transform=True)
ensemble1.fit(X_train_val, y_train_val)
pickle.dump(ensemble1, open(model_dir+ 'Ensemble1' + '.pkl', 'wb'))
joblib.dump(ensemble1, model_dir+ 'Ensemble1' + '.sav')

estimators=[('gradient_boost_classifier', gradient_boost_model), ('ada_boost', ada_boost_model), ('extra_tree', extra_tree_model)]#, ('decision_tree', dtree_model)]#('cat_boost', cat_boost_model)]#, ('decision_tree', dtree_model)]
#create our voting classifier, inputting our models
ensemble2 = VotingClassifier(estimators, voting='hard',  flatten_transform=True)
ensemble2.fit(X_train_val, y_train_val)
pickle.dump(ensemble2, open(model_dir+ 'Ensemble2' + '.pkl', 'wb'))
joblib.dump(ensemble2, model_dir+ 'Ensemble2' + '.sav')