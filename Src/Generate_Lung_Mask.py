from lungmask import mask
import SimpleITK as sitk
import matplotlib.pyplot as plt
import skimage
import os
import pdb
import numpy as np
import cv2
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import math
from scipy.io import savemat
from utils import *
import sys
import copy

def perform_maskUnetandResizeslices(patients_path, patient, outpath, temp_dir):
	
	
	final_path = patients_path + '/' +  patient
	files = os.listdir(final_path)
	files_sort = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
	slices = [sitk.ReadImage(final_path + '/' + f) for f in files_sort]
	total_num_slices = len(slices)
	Images = {}
	Mask_patient = {}
	final_segmentation = {}
	counter = 0
	
	
	for i, file in tqdm(enumerate(files_sort), desc = 'PatientName:'+ patient):
		input_image = sitk.ReadImage(final_path + '/' + file)
		
		if (len(np.unique(input_image)) == 1):
			continue
		
		Images[counter] = sitk.GetArrayFromImage(slices[i])
		segmentation = mask.apply(input_image)
		bin_img = copy.deepcopy(segmentation[0])
		bin_img[bin_img==2] = 1
		final_segmentation[counter] = bin_img * 255
		Mask_patient[counter] = segmentation[0]
		
		counter = counter + 1


	if (len(final_segmentation) != 0):
		saveDir = temp_dir + str(patient)
		if not os.path.exists(saveDir):
			os.makedirs(saveDir)
		

		Mask_array = np.stack([final_segmentation[i] for i in range(0,len(final_segmentation))])
		Image_array = np.stack([Images[i] for i in range(0,len(Images))])
		Mask = np.stack([Mask_patient[i] for i in range(0,len(Mask_patient))])
		Mask_array[Mask_array[:,:,:]==255] = 1
		SegmentedImage = Mask_array * Image_array

		for i in range(0,len(final_segmentation)):
			filename_image = saveDir + '/image_'+str(i)+'.jpg'
			filename_mask = saveDir + '/mask_'+str(i)+'.jpg'
			filename_seg_image = saveDir + '/seg_image_'+str(i)+'.jpg'
			cv2.imwrite(filename_image, Image_array[i])
			cv2.imwrite(filename_mask, final_segmentation[i])
			cv2.imwrite(filename_seg_image, SegmentedImage[i])

		
		saveDir = outpath + str(patient)
		if not os.path.exists(saveDir):
			os.makedirs(saveDir)

		### Saving Generated UNET Mask
		dest_file = saveDir + '/mask.mat'
		savemat(dest_file, {"data":Mask})
		
		### Saving original images
		dest_file = saveDir + '/image.mat'
		savemat(dest_file, {"data":Image_array})
	# if len(Image_array) >= 1:
	# 	return Mask, Image_array

