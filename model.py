import os
import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D

import matplotlib.pyplot as plt
        
def process_image(image):
    return image

steering_angles = []
car_images = []
#steering_angles = np.ndarray((0, 1))
#car_images = np.ndarray((0, 160, 320, 3))
def readAndPreprocessData(csv_file):
    print("Reading. " + os.path.dirname(csv_file))
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            path = "" # fill in the path to your training IMG directory
            img_center = process_image(np.asarray(cv2.imread(path + row[0])))
            img_left = process_image(np.asarray(cv2.imread(path + row[1])))
            img_right = process_image(np.asarray(cv2.imread(path + row[2])))

            # add images and angles to data set
            car_images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_center, steering_left, steering_right])

config = tf.ConfigProto()
#config.log_device_placement = True 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
                                    
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

def model_old():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3))) 
    model.add(Cropping2D(cropping=((70, 25),(0, 0))))
    #model.add(Lambda(lambda x: x / 2))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def model_nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3))) 
    model.add(Cropping2D(cropping=((70, 25),(0, 0))))
    #model.add(Lambda(lambda x: x / 2))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

reset_model = False
reset_model = True
#readAndPreprocessData('data1/driving_log.csv')
readAndPreprocessData('data3/driving_log.csv')
readAndPreprocessData('data4/driving_log.csv')
X_train = np.array(car_images)
y_train = np.array(steering_angles)

model_name = 'model.h5'

if reset_model:
    model = model_nvidia()
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs=10)
else:  
    model = load_model(model_name)
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs=10)

model.save(model_name)