# Gesture Detector Model - Part 2

## Author
Jiahui Li\
jiahui15@asu.edu

## Purpose
This project is to fulfill ASU MCS 535 class curriculum.

## Background
This is part 2 of the project. In part one, we developed an Android application that allows the use to watch 17 given hand gestures, including number 0 to 9, FanOn, FanOff, Increase Fan Speed, Decrease Fan Speed, LightOn, LightOff, and SetThermo. Then the user will be ask to record three 5 seconds videos for each of the gestures. The videos were save to a server by using Flask api.
This part 2 of the project is to develop a RESTful application for the SmartHome gestures classification by using a pre-trained. The user's recorded video from project 1 are used as labeled training gestures and given gesture videos as used as unlabeled testing gestures, both are first extracted for a single frame, and the frame will be analyzed by the with a pre-trained Tensorflow model. The unlabeled testing gestures will be classified by comparing the cosine similarity.

## Requirement
!!!Please Download pre-train model from: [Google Drive](https://drive.google.com/file/d/15jV8Czn4UZn0MIgsJuTVGtIl6IPamSpe/view?usp=sharing)!!!

Be sure [python](https://www.python.org/) is up to date!\
The following python library must be install prior to running the program:
* [Tensorflow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)

## Run
Use main.py to run the program
```bash
python main.py
```

## File/Folder
All files and folder must to be in the same working directory/folder.\
This project contains 4 files and 2 folders:
* main.py
    * program's main driver. Use this file to run the program
* cnn.h5* [Google Drive Download Link](https://drive.google.com/file/d/15jV8Czn4UZn0MIgsJuTVGtIl6IPamSpe/view?usp=sharing)
    * Pre-trained Tensorflow model
* frameextractor.py*
    * To extract a single frame from video
* handshape_feature_extractor.py*
    * To extract handshape feature from image by using cnn.h5 Tensorflow model
* test*
    * Unlabeled testing video
* traindata
    * Labeled training video recoded by user with application from project part 1

## Output
This project will creates 2 folders and 1 file as being ran:
* extracted_test_image
    * This folder contains all the extracted unlabeled testing video frame as .png image
* extracted_training_image
    * This folder contains all the extracted labeled training video frame as .png image
* Result.csv
    * This single column .csv file contain the result for classification of the testing video
    * The data is sorted by the order of the extracted_training_image

## Credit
Files labeled with (*) in the [File/Folder](##File/Folder) sections are provided by Arizona State University MCS 535
