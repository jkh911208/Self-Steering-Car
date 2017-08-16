#!/usr/bin/env python 

import os
import tensorflow as tf
import model
import params
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,300) # set height
cap.set(4,200) # set width

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "./weight/SSC_epoch_14_LR_0.0001.model")

def get_frame():
	# read the frame from webcam
	_, frame = cap.read()
	tmp = cv2.resize(frame,(0,0),fx=0.8, fy=0.8)
	tmp = cv2.flip(tmp,0)
	tmp = cv2.flip(tmp,1)
	if params.img_channels != 3:
		tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	img = (tmp[70:-5,::]).reshape(params.img_height, params.img_width, params.img_channels)

	return img, frame

def get_angle():
    while(True):
	# get one can data
        can_data = str(bus.recv())
	# check it is steering angle, if yes then stop the loop
        if(can_data.find("ID: 0025") > 0):
            break

    # change the can data (HEX) to numerical data
	tmp = can_data
	hex_data = tmp[-23:-21] + tmp[-20:-18]
	hex_decimal = tmp[-3:-1]
	int_data = int(hex_data, 16)
	int_decimal = int(hex_decimal, 16) / 256

			# if the steering wheel angle in in right to the center
	if(int_data > 550):
		int_data = int_data - 4096
		int_decimal = 1 - int_decimal 
		final_data = int_data - int_decimal
	else:
		# put the int and the decimal together
		final_data = int_data + int_decimal

    return final_data

def main():
	# discard first 40 frames to give time for webcam to get the proper exposure
	for i in range(40):
			get_angle()
			get_frame()
			print(i, "discarded")

	while True:
		angle = get_angle()
		img, frame = get_frame()

		deg = model.y.eval(feed_dict={model.x: [img], model.keep_prob: 1.0})[0][0]
		difference = ((final_data+16)-(deg+16))/(final_data+16)
		if difference < 0:
			difference = difference * -1
		difference = difference * 100

		predicted = "predicted : " + str(deg)
		actual = "actual : " + str(final_data)
		percent = "% difference : " + str(difference)

		cv2.putText(image,predicted,(10,30), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
		cv2.putText(image,actual,(50,30), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
		cv2.putText(image,percent,(70,30), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
		
		cv2.imshow('frame', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		

main()

cap.release()
cv2.destroyAllWindows()


			    
