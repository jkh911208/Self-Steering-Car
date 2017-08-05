#!/usr/bin/env python 

import params
import cv2
import numpy as np

def train_data(file_name):
	file_location = "/raw_data/" + file_name
	loaded_data = np.load(file_location)
	
	X = []
	Y = []
	temp_x = []

	# change the saved data into the form that both human and NN can understand
	for data in loaded_data:
		# flip the frame because the webcam is mounted upside down on the front windsheild, and cut out the sky portion of the image the size became 256*66
		tmp = cv2.flip(data[0],0)
		tmp = cv2.flip(tmp,1)
		if params.img_channels != 3:
			tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
		frame = (tmp[70:-5,::]).reshape(params.img_height,params.img_width,1)

		# change the can data (HEX) to numerical data
		tmp = data[1]
		hex_data = tmp[-23:-21] + tmp[-20:-18]
		hex_decimal = tmp[-3:-1]
		int_data = int(hex_data, 16)
		int_decimal = int(hex_decimal, 16) / 256

		# if the steering wheel angle in in right to the center
		if(int_data > 550):
			int_data = int_data - 4095
			int_decimal = 1 - int_decimal 
			final_data = int_data - int_decimal 
		else:
			# put the int and the decimal together
			final_data = int_data + int_decimal

		if len(temp_x) == 0:
			temp_x = frame
		elif temp_x.shape[0] < 66*4:
			temp_x = np.concatenate((temp_x, frame), axis=0)
		elif temp_x.shape[0] == 66*4:
			temp_x = np.concatenate((temp_x, frame), axis=0)
			X.append([temp_x])
			Y.append([final_data])
		else:
			temp_x = temp_x[66:,::,::]
			temp_x = np.concatenate((temp_x, frame), axis=0)
			X.append([temp_x])
			Y.append([final_data])

	X = X[50:]
	Y = Y[50:]
	X = np.array(X).reshape([-1, params.network_height, params.img_width, params.img_channels])
	Y = np.array(Y).reshape([-1,1])

	return X, Y

	#finishing making the training data
