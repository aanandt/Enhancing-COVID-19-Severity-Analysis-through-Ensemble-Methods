import glob
import csv
from utils import *
import  cv2 as cv
from tqdm import tqdm
from scipy import ndimage 
import os  
from scipy.io import loadmat
import copy 
import pandas as pd
import pdb 
import math
from tqdm import tqdm 
from joblib import Parallel, delayed
import multiprocessing
from sklearn.metrics import f1_score



def proposed_pipeline(curr_mask, ct_img1, slice_num, Masks, save_dir):

	curr_sum = ((np.sum(curr_mask[curr_mask==2])/2) + np.sum(curr_mask[curr_mask==1]))
	left_sum = np.sum(curr_mask[curr_mask==1])
	right_sum = (np.sum(curr_mask[curr_mask==2])/2)

	if ( (left_sum == 0) or (right_sum == 0) or ((left_sum/right_sum) <= 0.75 and left_sum > 15000) or ((right_sum/left_sum) <= 0.75 and right_sum > 15000)):
		template = find_template(curr_mask, Masks, slice_num)
		curr_mask = copy.deepcopy(template)
		Masks[slice_num,:,:] = template


	

	# global thresholding
	ret1,mask1 = cv.threshold(ct_img1,90,255,cv.THRESH_BINARY)
	ct_img = np.multiply(mask1, ct_img1)
	#cv.imwrite(out_dir+'/threshold_based_binary_img'+ str(slice_index) +'.png', ct_img)

	### Histogram Hyperbolization
	blur = cv.GaussianBlur(ct_img,(5,5),0)
	try:
  		ct_hyper =  HistHyper(blur)#bpdfhe(blur)#HistHyper(blur)
	except:
  		ct_hyper =  ct_img #HistHyper(blur)#bpdfhe(blur)#HistHyper(blur)
  		print("============="+save_dir + "==================")
	image = ct_hyper.astype("uint8")
	image[image>=image[0][0]] = 0
	#cv.imwrite(out_dir+'/blurred_hyper_image'+ str(slice_index) +'.png', image)

	### Identify the binary mask for the further analysis
	ret, mask2 = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
	#cv.imwrite(out_dir+'/blurred_hyper_image_OTSU'+ str(slice_index) +'.png', (mask2))

	filterSize =(3,3)
	kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
	opening = cv.morphologyEx(mask2, cv.MORPH_OPEN, kernel)
	mask3 = copy.deepcopy(opening)
	mask3[mask3 == 0] = 0
	mask3[mask3 == 255] = 1

	## Removal of small connected component
	mask3 = find_connected_component(mask3)
	result_img = np.multiply(mask3,  ct_img)
	#cv.imwrite(out_dir+'/blurred_hyper_image_OTSU_areaopen'+ str(slice_index) +'.png', result_img)


	#### Vessel removal 
	filterSize =(9, 9)
	kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
	tophat_img = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)
	ret, mask4 = cv.threshold(tophat_img, 0, 255, cv.THRESH_OTSU)
	mask4[mask4==255] = 1
	vessel_filter = mask4 * ct_img
	temp_vessel = copy.deepcopy(vessel_filter)
	temp_vessel[temp_vessel > 0] = 1
	#cv.imwrite(out_dir+'/vessel_filter'+ str(slice_index) +'.png', vessel_filter)

	mask5 = find_connected_component(temp_vessel)
	#cv.imwrite(out_dir+'/vessel_filter_finetuned'+ str(slice_index) +'.png', (mask5*255))

	if(np.sum(mask5)/curr_sum <= 0.25):
		#im_result[im_result == 255] = 1
		blood_vessel_enhanced = result_img * mask5
		infection_mask = result_img - blood_vessel_enhanced
		result_img = copy.deepcopy(infection_mask)

	#cv.imwrite(out_dir+'/blurred_hyper_image_OTSU_areaopen_vessel_filter'+ str(slice_index) +'.png', result_img)

	### Filling the holes where the blood vessel removal creates holes in the infection region
	result_new = ndimage.binary_fill_holes(result_img)
	filterSize =(2, 2)
	kernel = cv.getStructuringElement(cv.MORPH_RECT, filterSize)
	result_new = result_new.astype("uint8")
	result_new = cv.dilate(result_new, kernel)
	infection_region = result_new * result_img
	
	

	#infection_region = copy.deepcopy(result_img)
	filename = os.path.join(save_dir, 'infection_mask'+ str(slice_num) +'.png')
	cv.imwrite(filename, infection_region)
	#cv.imwrite(out_dir+'/blurred_hyper_image_OTSU_areaopen_vessel_filter_fillholes_dilated'+ str(slice_index) +'.png', infection_region)
	
	infection_region[infection_region > 0] = 1
	infection_region = infection_region.astype("uint8")
	### Finding the severity score based on the infection mask
	final_score, features = find_weighted_score(curr_mask, infection_region)
	

	return final_score, features



def Extract_features(path, patient, preprocessed_path, threshold):#Extract_features(Masks, images, patient, preprocessed_path, threshold):

	Masks = loadmat(os.path.join(path, patient, 'mask.mat'))['data']
	images = loadmat(os.path.join(path, patient, 'image.mat'))['data']
	save_dir = os.path.join(preprocessed_path, patient)
	make_dirs(save_dir)
	
	temp_masks = copy.deepcopy(Masks)
	temp_masks[temp_masks==2] = 1
	segmented_images = images * temp_masks

	slicewise_infection_percentage_threshold_slice_num = []
	slicewise_infection_percentage_threshold_lung_mask = []

	total_num_slices = len(Masks)
	feature_list = []
	for slice_num, seg_file in tqdm(enumerate(segmented_images), desc= 'Processing patient:'+patient):
		#pdb.set_trace()
	
		curr_mask = copy.deepcopy(Masks[slice_num,:,:])
		curr_sum = ((np.sum(curr_mask[curr_mask==2])/2) + np.sum(curr_mask[curr_mask==1]))
		ct_img1 = segmented_images[slice_num,:,:]
		
		if (((slice_num >= (total_num_slices/3) and slice_num <= ((2 * total_num_slices)/3)) and curr_sum > 9000) or (curr_sum/ (512 * 512)) >= threshold):
		
			final_score, features = proposed_pipeline(curr_mask, ct_img1, slice_num, Masks,save_dir)
			slicewise_infection_percentage_threshold_slice_num.append(final_score)
			feature_list.append(features)
		

		if(len(slicewise_infection_percentage_threshold_slice_num) <= 50):
			severity_score_slice_threshold = np.average(slicewise_infection_percentage_threshold_slice_num)
			
		else:
			slicewise_infection_percentage_threshold_slice_num[slicewise_infection_percentage_threshold_slice_num == 0] = np.nan
			severity_score_slice_threshold = np.nanmean(slicewise_infection_percentage_threshold_slice_num)

		
		severity_score = severity_score_slice_threshold
	return severity_score, feature_list

