2017 Summer Personal Project on Deep Learning with Tensorflow

1. Prepare Hardware
2. Collect Data
3. Design Model
4. Prepare the Data
5. Train the Model
6. Test the Model


<1> Prepare Hardware

	I used Nvidia Jetson TX2 single board computer for both collecting the data and training the network. 

	Installed a cheap (<$20) usb webcam on the front windsheild, Since the CNN do not require the high resolution, so it was a decent option for me. For the better image quility you can use some more expansive webcam such as c920, which is widely used for the robot.

	2016 Toyota Camry. I used can bus in my car to get the steering wheel angle data. In order to do so, I had to connect the extra cable from back of the radio head (canH, canL) and wired a ACC power (12v, DC) for the main power source for the Jetson TX2.

<2> Collect Data

<3> Design Model

	Input is the video frame from the usb webcam and the output is the steering wheel angle from the can bus. 

	Used googLeNEt for the main network. This is the linear regression problem, so the activation function for the last layer is "linear" and the cost function I used "Mean Squre Error". Used "Adam" optimizer and "1e-2" for the LR.

<4> Prepare the Data

	For the image file it was easy. convert it to the gray image and cut out the sky portion of the image

	For the steering angle data, I had to reverse engineer the can bus data from Toyota. I found that they are using the signed and the fixed point type. The first two bytes were representing the int value before the decimal and the last byte was representing the numbers after the decimal. I think this is not the best way to decode the can bus data, but I believe this is enough accurate for the personal project. 

<5> Train the Model

<6> Test the Model