#!/usr/bin/env python 

import os
import tensorflow as tf
import model
import params
import time
import cv2
import numpy as np

def train_data(file)
	file_location = "/raw_data/" + file
	loaded_data = np.load(file_location)

	# change the saved data into the form that both human and NN can understand
	for data in loaded_data:
		# flip the frame because the webcam is mounted upside down on the front windsheild, and cut out the sky portion of the image the size became 256*66
		tmp = cv2.flip(data[0],0)
		tmp = cv2.flip(tmp,1)
		if params.img_channels != 3:
			tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
		data[0] = (tmp[70:-5,::]).reshape(params.img_height,params.img_width,params.img_channels)

		# change the can data (HEX) to numerical data
		tmp = data[1]
		hex_data = tmp[-23:-21] + tmp[-20:-18]
		hex_decimal = tmp[-3:-1]
		int_data = int(hex_data, 16)
		int_decimal = int(hex_decimal, 16)

		# if the steering wheel angle in in right to the center
		if(int_data > 550):
			int_data = int_data - 4096
			int_decimal = 256 - int_decimal 

		# put the int and the decimal together
		num_in_string = str(int_data) + "." + str(int_decimal)
		final_data = float(num_in_string)
		data[1] = final_data
		#print(final_data)
	#did cut out the first few frames because the webcam need time to adjust the exposure
	# but seems like still the white out frames exist

	loaded_data = loaded_data[30:]
	train_X = np.array([i[0] for i in loaded_data]).reshape([-1, params.img_height,params.img_width,params.img_channels])
	train_Y = np.array([i[1] for i in loaded_data]).reshape([-1,1])

	#finishing making the training data
