# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_plot.png "Model Visualization"
[image2]: ./examples/center_2019_08_11_21_27_06_924.jpg "Centerline Drive"
[image3]: ./examples/center_2019_08_15_10_15_20_414.jpg "Recovery Image 1"
[image4]: ./examples/center_2019_08_15_10_15_22_387.jpg "Recovery Image 2"
[image5]: ./examples/center_2019_08_15_10_15_23_265.jpg "Recovery Image 3"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* **video_data.mp4** showing the one round of autonomous driving

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to stick to the models used in the lecture videos when getting teached on how to use the simulator and on improving the model. 

To improve the quality of the training all images pass a normalization layer and a cropping layer. The cropping layer cuts distracting environment feature at the top and the bottom of the picture.  

The first reasonable network had a LeNet-like architecture (model.py lines 52-63). A set of convolution - relu activation - pooling layers followed by a few fully connected layers has already shown its strength in image classification. 

Since the performance of the LeNet-like architecture was insufficient, a neural network model invented by nvidia was implemented. After a training on a reasonable data set this model was able to steer the simulation vehicle around the track.

#### 2. Final Model Architecture

The final model architecture (model.py lines 67-80) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture

| Layer (type)     			|     Output Shape		|    Param #	| 
|:-------------------------:|:---------------------:|:-------------:|
| lambda_2 (Lambda) 		| (None, 160, 320, 3)	| 0				|      
| cropping2d_2 (Cropping2D)	| (None, 65, 320, 3)	| 0				|         
| conv2d_6 (Conv2D)			| (None, 31, 158, 24)	| 1824			|      
| conv2d_7 (Conv2D)			| (None, 14, 77, 36)	| 21636			|     
| conv2d_8 (Conv2D)			| (None, 5, 37, 48)		| 43248			|    
| conv2d_9 (Conv2D)			| (None, 3, 35, 64)		| 27712			|     
| conv2d_10 (Conv2D)		| (None, 1, 33, 64)		| 36928			|     
| flatten_2 (Flatten)		| (None, 2112)			| 0				|         
| dense_5 (Dense)			| (None, 100) 			| 211300		|    
| dense_6 (Dense)			| (None, 50)			| 5050			|     
| dense_7 (Dense)			| (None, 10)			| 510			|
| dense_8 (Dense)			| (None, 1)				| 11			|

Total params: 348,219

Trainable params: 348,219

Non-trainable params: 0
_________________________________________________________________

![alt text][image1]

#### 3. Attempts to reduce overfitting in the model
The model is created in model.py lines 67-80. 
The model does not need dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 94-96). The training data was shuffeled each run. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 4. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 95). Improvement was made by applying a small correction to the side cameras steering angle twords the image center (model.py lines 29-31). The value of 0.2 for the correction turned out to work quite well.

#### 5. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving varying direction, recovering from the left and right sides of the road


#### 6. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to find its way back to the center line. These images show what a recovery looks like starting from the road boundary:

![alt text][image3]
![alt text][image4]
![alt text][image5]


After the collection process, I had 13569 number of data points. I then preprocessed this data by applying normalization and image cropping as explained above.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by low mean square error loss and a sufficient prediction perfomance of the model.
