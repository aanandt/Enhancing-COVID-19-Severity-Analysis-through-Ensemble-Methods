import os
from tqdm import tqdm
import pandas as pd
from utils import *

from Generate_Lung_Mask import *
from Extract_feature_vector import *

gt_filename = '../Dataset/.xlsx' # Path to ground truth file in .xlsx format
path = '../Temp_dir/Validation/Masks_and_Images/' #Path to train-dataset
mask_image_path = '../Temp_dir/Validation/Masks_and_Images/' #Path to lung mask and original ct images
temp_dir = '../Temp_dir/Validation/All_files/'   #Path to store the images, mask, and segmented images

preprocessed_path = '../Temp_dir/Validation/Preprocessed_Images/' # Path to the infection segementation
feat_filename = 'features_Validation.npy' # Validation feature filename
feat_dir = '../Features/Validation/' # Path to Validation feature directory
result_dir = '../Results/Models/'
model_dir = '../Models/saved_models1/'

if not os.path.exists(feat_dir):
	os.makedirs(feat_dir)
if not os.path.exists(result_dir):
	os.makedirs(result_dir)


df = pd.read_excel(gt_filename)
val_Y= list(map(int, df['Category']-1))
y_test = np.asarray(val_Y, dtype=np.int64)
test_patients = list(df['Name'])


Percentage_patientwise_infection = {}
Patientwise_infection_features = []
severity_predictions = []
patients = os.listdir(path)
pdb.set_trace()
sorted_patients = sorted(patients, key=lambda x: int(x.split('.')[0].split('_')[2]))
lung_threshold = 0.07
counter = 0

# for patient in tqdm(sorted_patients, desc = 'Testing patient\'s CT_scans'):
# 	perform_maskUnetandResizeslices(path, patient, mask_image_path, temp_dir)
	
# 	counter = counter + 1
# pdb.set_trace()
# num_cores = multiprocessing.cpu_count() - 5
# infection_rate = Parallel(n_jobs=num_cores)(delayed(Extract_features)(mask_image_path, patient, preprocessed_path, lung_threshold)for patient in (sorted_patients))

# for i, pair_value in enumerate(infection_rate):
# 	Patientwise_infection_features.append(pair_value[1])
# 	score = find_final_severity_score(pair_value[0])
# 	severity_predictions.append(score)
# pdb.set_trace()

#np.save(feat_dir +feat_filename, np.array(Patientwise_infection_features, dtype=object), allow_pickle=True)
features_list = (np.load(os.path.join(feat_dir +feat_filename), allow_pickle=True))
X_test = Create_input_features(features_list)
#X_test = Create_input_features(Patientwise_infection_features)



# ### Evaluating the models

##### Weighted average method
# cm = confusion_matrix(y_test, severity_predictions)
# print(cm)
# ax = plt.subplot()
# sns.set(font_scale=3.0)
# sns.heatmap(cm, annot=True, cmap='Blues', cbar=False)
# title_font = {'size':'30'}  # Adjust to fit
# ax.set_title('Confusion Matrix', fontdict=title_font);
# ax.tick_params(axis='both', which='major', labelsize=30)  # Adjust to fit
# ax.xaxis.set_ticklabels(['Mi' , 'Mo', 'Se', 'Cr']);
# ax.yaxis.set_ticklabels(['Mi' , 'Mo', 'Se', 'Cr']);
# plt.show()
# plt.savefig(result_dir+'WAM_confusionmatrix.png')
# plt.close()
# print(classification_report(y_test, severity_predictions,target_names=['Mild', 'Moderate', 'Severe', 'Critical']))

### Evaluating ML models
lgr_model = pickle.load(open(model_dir + 'Logistic_regression.pkl', 'rb'))
Evaluate_model('Logistic_regression', lgr_model,  X_test, y_test, result_dir)

knn_model = pickle.load(open(model_dir + 'kNN.pkl', 'rb'))
Evaluate_model('kNN', knn_model, X_test, y_test, result_dir)

naive_Bayes_model = pickle.load(open(model_dir + 'Naive_bayes.pkl', 'rb'))
Evaluate_model('Naive_bayes', naive_Bayes_model, X_test, y_test, result_dir)

svm_model = pickle.load(open(model_dir + 'SVM.pkl', 'rb'))
Evaluate_model('SVM', svm_model, X_test, y_test, result_dir)

dtree_model = pickle.load(open(model_dir + 'Decision_tree.pkl', 'rb'))
Evaluate_model('Decision_tree', dtree_model,  X_test, y_test, result_dir)

nn_model = pickle.load(open(model_dir + 'Neural Network.pkl', 'rb'))
Evaluate_model('Neural_network', nn_model, X_test, y_test, result_dir)

rf_model = pickle.load(open(model_dir + 'Random_forest.pkl', 'rb'))
Evaluate_model('Random_forest', rf_model, X_test, y_test, result_dir)

xgb_model = pickle.load(open(model_dir + '/XG_boost.pkl', 'rb'))
Evaluate_model('XG_boost', xgb_model, X_test, y_test, result_dir)

ada_boost_model = pickle.load(open(model_dir + 'AdaBoostClassifier.pkl', 'rb'))
Evaluate_model('AdaBoostClassifier', ada_boost_model, X_test, y_test, result_dir)

extra_tree_model = pickle.load(open(model_dir + 'ExtraTreesClassifier.pkl', 'rb'))
Evaluate_model('ExtraTreesClassifier', extra_tree_model, X_test, y_test, result_dir)

gradient_tree_model = pickle.load(open(model_dir + 'GradientBoostingClassifier.pkl', 'rb'))
Evaluate_model('GradientBoostingClassifier', gradient_tree_model, X_test, y_test, result_dir)

### RF, SVM, and ERT
Ensemble = pickle.load(open(model_dir + 'Ensemble.pkl', 'rb'))
Evaluate_model('ensemble', Ensemble, X_test, y_test, result_dir)

### XGboost, SVM, and ERT
Ensemble1 = pickle.load(open(model_dir + 'Ensemble1.pkl', 'rb'))
Evaluate_model('Ensemble1', Ensemble1, X_test, y_test, result_dir)

### Gboost, Adaboost, and ERT
Ensemble2 = pickle.load(open(model_dir + 'Ensemble2.pkl', 'rb'))
Evaluate_model('Ensemble2', Ensemble2, X_test, y_test, result_dir)
