# **Behavioral Cloning**


### Introduction:
This approach uses applied deep neural networks and convolutional neural networks (CNNs) to clone driving behavior by training,
validating and testing a model using Keras. The model will output a steering angle to an autonomous vehicle. Data collection was
accomplished using a simulator which allows for steering a car around a track. Using the collected image data and steering angles,
a neural network shall be trained on the data. Finally this model shall be applied to drive the car autonomously around the track
which shall serve as a validation of the neural network.


---

**Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of which characterized good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

Project files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it
contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Trained on a AWS GPU (4G), my model consists of a convolution neural network with 5 convolutional layers, and filter sizes of 3x3 and 2X2 respectively.
Network depths range between 24 and 64 (model.py lines 81-85)

The model includes RELU layers to introduce nonlinearity (code lines 81-85), and the data was normalized in the model using a Keras lambda layer (code line 79).

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
    ____________________________________________________________________________________________________
    cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]
    ____________________________________________________________________________________________________
    convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]
    ____________________________________________________________________________________________________
    convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]
    ____________________________________________________________________________________________________
    convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]
    ____________________________________________________________________________________________________
    convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]
    ____________________________________________________________________________________________________
    convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]
    ____________________________________________________________________________________________________
    flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]
    ____________________________________________________________________________________________________
    dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]
    ____________________________________________________________________________________________________
    dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]
    ____________________________________________________________________________________________________
    dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]
    ____________________________________________________________________________________________________
    dropout_3 (Dropout)              (None, 10)            0           dense_3[0][0]
    ____________________________________________________________________________________________________
    dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]
    ====================================================================================================

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 88,90,92).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 67-68). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.  Data collection was time-consuming and cumbersome as it involved several trials using the provided simulator.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving.  
The data collection strategy I used was as follows:
  i.  3 laps forward direction
  ii. 2 laps reverse direction
  iii. 2 laps steering right to left and vice versa.  

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to seek a well established CNN and repurpose it for behavioral cloning.  

I used a convolution neural network model which is similar to (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)  
I thought this model might be appropriate because it is well known and has been tested in the real-world scenario.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set.
This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it has dropout layers (model.py lines:  88,90,92)


The final step was to run the simulator to see how well the car was driving around track one. I found that the result was very good based on the data I personally collected (see video output file)
The car successfully completed the entire first lap without going off the road and always staying in between the yellow lines. The car accomplished the goal of being able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 78-93) consisted of a convolution neural network with the following layers and layer sizes:
5 CNNs, 5x5, 3x3

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][https://github.com/knasim/Behavioral-Cloning/blob/master/images/2018_03_04_22_37_09_528.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to handle these driving behaviors.  Then I repeated this process on by recording 3 laps in the reverse direction.

To augment the data sat, I also flipped images and angles thinking that this would help with turn bias.   
For example, here is an image that has then been flipped:

![alt text](https://github.com/knasim/Behavioral-Cloning/blob/master/images/2018_03_04_22_37_09_528.jpg)
![alt text](https://github.com/knasim/Behavioral-Cloning/blob/master/images/2018_03_04_22_37_09_529.jpg)


After the collection process, I ended with a CSV file just under 5MB.  I managed to collect a total of 12,349 data points.
I then preprocessed this data by using model.py

I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
For me the ideal number of epochs was  5 and I used  batch size of 32.  I used an adam optimizer so that manually training
the learning rate wasn't necessary.
