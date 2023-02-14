# -*- coding: utf-8 -*-
"""
Template Created on Thu Jan 28 00:44:25 2021 by chakati
Modified on October 7, 2022

@Modified: Jiahui Li 
"""
## import the handfeature extractor class
import csv
import cv2
import numpy as np
import os
import tensorflow as tf
import frameextractor as fet
from numpy import dot
from numpy.linalg import norm
from handshape_feature_extractor import HandShapeFeatureExtractor as hfe

name_dict = {"FANDOWN":"10", "FANOFF":"12", "FANON":"11", 
"FANUP":"13", "LIGHTOFF":"14", "LIGHTON":"15", "NUMEIGHT":"8", 
"NUMFIVE":"5", "NUMFOUR":"4", "NUMNINE":"9", "NUMONE":"1", 
"NUMSEVEN":"7", "NUMSIX":"6", "NUMTHREE":"3", "NUMTWO":"2", 
"NUMZERO":"0", "SETTHERMO":"16"}

#For percentage calculation use only
"""test_name_dict= {"0.mp4": "0", "1.mp4": "1", "2.mp4": "2", 
"3.mp4": "3", "4.mp4": "4", "5.mp4": "5", "6.mp4": "6", "7.mp4": "7", 
"8.mp4": "8", "9.mp4": "9", "DecreaseFanSpeed.mp4": "10", "DecereaseFanSpeed.mp4": "10",
"FanOff.mp4":"12", "FanOn.mp4":"11", "IncreaseFanSpeed.mp4":"13", 
"LightOff.mp4": "14", "LightOn.mp4":"15", "SetThermo.mp4":"16"}"""
test_name_dict2= {}
image_name = {}
train_dict= {}
test_dict = {}
prediction= {}

new_feature_extractor = hfe()
cwd = os.getcwd()
# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
video_folder = "traindata"
image_folder = "extracted_training_image"
file_type = ".mov"
directory_path = os.path.join(cwd,video_folder)
count = 0
image_path = os.path.join(cwd,image_folder)
for file in os.listdir(directory_path):
	if file.endswith(file_type):
		full_path = os.path.join(directory_path, file)
		image_name[fet.frameExtractor(full_path, image_path, count)
		.split("/")[1]] = file.split("_")[0]
		count = count+1
# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video
video_folder = "test"
image_folder = "extracted_test_image"
file_type = ".mp4"
directory_path = os.path.join(cwd,video_folder)
count = 0
image_path = os.path.join(cwd,image_folder)
for file in os.listdir(directory_path):
	if file.endswith(file_type):
		full_path = os.path.join(directory_path, file)
		test_name_dict2[fet.frameExtractor(full_path, image_path, count)
		.split("/")[1]] = file.split("-")[-1]
		count = count+1
# =============================================================================
# Extract training gesture
# =============================================================================
image_folder = "extracted_training_image"
file_type = ".png"
image_path = os.path.join(cwd,image_folder)
for image in os.listdir(image_path):
	if image.endswith(file_type):
		full_image_path = os.path.join(image_path, image)
		im = cv2.imread(full_image_path)

		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		output = new_feature_extractor.extract_feature(gray)

		key = name_dict[image_name[image]]
		if key in train_dict.keys():
			train_dict[key] = (output + train_dict[key])
		else:
			train_dict[key] = output
# =============================================================================
# Extract test gesture
# =============================================================================
image_folder = "extracted_test_image"
file_type = ".png"
image_path = os.path.join(cwd,image_folder)
for image in os.listdir(image_path):
	if image.endswith(file_type):
		full_image_path = os.path.join(image_path, image)
		im = cv2.rotate(cv2.imread(full_image_path), cv2.ROTATE_180)

		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		output = new_feature_extractor.extract_feature(gray)

		key = image
		test_dict[key] = output
# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
for item in test_dict:
	a = np.squeeze(np.asarray(test_dict[item]))
	for item2 in train_dict:
		b = np.squeeze(np.asarray(train_dict[item2]))
		cosine = np.dot(a,b)/(norm(test_dict[item])*norm(train_dict[item2]))
		if item in prediction.keys():
			if prediction[item][1] < cosine:
				prediction[item] = [item2, cosine]
		else:
			prediction[item] = [item2, cosine]
# =============================================================================
# Writing to csv file
# =============================================================================
with open('Results.csv', 'w') as f:
	write = csv.writer(f)
	for x in range(51):
		if x < 9:
			name = "0000" + str(x+1) +".png"
		else:
			name = "000" + str(x+1) +".png"
		number = prediction[name][0]
		write.writerow([number])

with open('Results.csv','r') as f:
    lines = f.readlines()
    line = lines[len(lines)-1]
    lines[len(lines)-1] = line.rstrip()
with open('Results.csv', 'w') as f:    
    f.writelines(lines)
# =============================================================================
# Print percentage
# =============================================================================
"""
percentage = 0
for item in prediction:
	if prediction[item][0] == test_name_dict[test_name_dict2[item]]:
		percentage = percentage+1
print(percentage/51*100)
"""













