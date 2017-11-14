# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run2.mp4 driving record video in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

nVidia model is employed as the model architecture.

The model consists of 5 convolutional layers. First 3 of them have filters of 5 * 5 and the depth is 24 ~ 48, the other 2 layers have filters of 3 * 3 and the depth is 64.

The model includes RELU layers to introduce nonlinearity (code lines 41 ~ 45), and the data is normalized to -0.5 ~ -0.5 in the model using a Keras lambda layer (code line 39). With Keras Cropping2D function, the image is also cropped the upper 50 pixels and lower 20 pixels (code line 40), which include non-interested pixels like trees, sky and car front cover. 

Normalization is really a good method. After introducing Normalization, the training loss falls down very much.

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in the first 2 full connection layers in order to reduce overfitting (lines 48, 50). The dropout factor is 0.5.

The model was trained and validated on data set provided by Udacity. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (line 156).

#### 4. Appropriate training data

Training data from Udacity was chosen to keep the vehicle driving on the road. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As I trained the model in the AWS EC2, It will take a while to download the model.h5 file to run the model in the simulator. A better choice may be testing the model in the EC2 with the model predict API. Therefore, the first thing I had done was to choose 2 images from the data set for the prediction. One image has  '-0.23' steering angle and the other '+0.367' steering angle. 

Sadly, during the fine tuning of the model, the training loss and validation loss can reached around 0.01. But the prediction of the '+' angle image was always not desirable, around 0.1 ~ 0.08 or worse, even when the predicted '-' angle reached -0.20. 

Since I had flipped all the images with the openCV in the generator, it was really weird to me that the '+' image can not get a result as with the '-' image. Moreover, when the prediction of both the image were not well, the trained model was able to drive autonomously around the track without leaving the road. 

Until the due time, still I haven't figure this out.

#### 2. Final Model Architecture

The final model architecture (model.py lines 38-52) is just the nVidia model.

#### 3. Creation of the Training Set & Training Process

While regarding the training set, the data set from Udacity is employed at last. Besides, I have also collected my own data. I have recorded 7 laps on track one.
* 4 laps in anti-clockwise using center lane driving
* 2 laps clockwise, and 
* 1 laps recovery from the side.

To augment the data set, I also flipped images. And images from the left and right cameras are also used with correction of 0.1. Correction 0.2 is a little bit too much for the track one.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 to avoid overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Writing in the end
As for this project, I spent a lot of time to tackle with the unprecisely prediction of the image, and wrong writing of correction in left and right cameras. The prediction problem is still not solved yet. But I guest i will never forget to '+' correction in left and '-' correction in right camera.




