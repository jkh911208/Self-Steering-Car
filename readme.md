# 2017 Summer Personal Machine Learning (Deep Learning) Project (Linear Regression)

## Inspiration:
Project was inspired by "Deeptesla" (https://github.com/lexfridman/deeptesla.git)

## Description:
Use TensorFlow to Inference the steering wheel angle only based on the single front image

# <1> Developing Environment

## (Hardware)
- Nvidia Jetson TX2 (256 Pascal CUDA Cores, 8GB shared RAM)
- Texas Instrument SN65HVD230 (Can bus transceiver)
- Toyota 2016 Camry LE
- Wires and Electricity (12V DC)
- Logitech Webcam (Cheapest possible from Micro Center)

## (Software)
- Tensorfow v1.0.1 
- Python3
- OpenCV 3.2
- Numpy

# <2> Collect Data

I drove my car by myself and at the same time collecting the data. It is possible to connect the keyboard to the Jetson TX2 and run the Python code, but I found that it is more convenient to use the serial port in the GPIO port and connect to my MacBook and run the script. 

# <3> Design Model

Network: Modified AlexNet to fit my model

Loss: Mean Squared Error

# <4> Prepare the Data
I resized the frame into size of [66,256,1] which I personally thought is good enough for the CNN. 

I could find that the Toyota uses the "ID : 0x025" as the steering wheel angle. I found that they are using the signed data, fixed point decimal. For more detail please check the code.

## (Balance Data)
Steering wheel usually don't change a lot in the real world driving environment. Therefore, the data was extremely imbalanced. I spend lots of time balancing the data.

# Lessons Learned:

## 1. Hardware
  CNN requires big amount of memory, if the dataset don't fit into the memory, then cut it down to smaller batch.
## 2. Network 
  Before I start this project I thought the network will make huge difference in linear regression problem, because it does make huge difference in categorical problem. However, I found that the linear regression problem doesn't hugely affected by the network
## 3. Data
  It was challenging to make the data into the dataset and balance the data prevent from biased weight.





### Steering angle of 15:

![img_7048](https://user-images.githubusercontent.com/17028674/29476442-a6a2d89c-8429-11e7-95a7-e0ba01185976.JPG)

### Steering angle of -15:

![img_0727](https://user-images.githubusercontent.com/17028674/29476443-a6cc2d14-8429-11e7-8b7f-1f77806bb011.JPG)
