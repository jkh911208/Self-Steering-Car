#!/usr/bin/env python 

import os
import params
import cv2
import numpy as np

# make the val data

def val_data():
	val_x = []
	val_y = []
	file_list = os.listdir('/raw_data/eval_data')
	for file in file_list:
		if file.endswith('.npy'):
			file_location = "/raw_data/eval_data/" + file
			loaded_data = np.load(file_location)
			
			X = []
			Y = []
			temp_x = []

			for data in loaded_data:
				tmp = cv2.flip(data[0],0)
				tmp = cv2.flip(tmp,1)
				if params.img_channels != 3:
					tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
				frame = (tmp[70:-5,::]).reshape(params.img_height, params.img_width, 1)

				# change the can data (HEX) to numerical data
				tmp = data[1]
				hex_data = tmp[-23:-21] + tmp[-20:-18]
				hex_decimal = tmp[-3:-1]
				int_data = int(hex_data, 16)
				int_decimal = int(hex_decimal, 16) / 256

				# if the steering wheel angle in right to the center
				if(int_data > 550):
					int_data = int_data - 4095
					int_decimal = 1 - int_decimal 
					final_data = int_data - int_decimal 
				else:
					# put the int and the decimal together
					final_data = int_data + int_decimal

				# stack up the image
				if len(temp_x) == 0:
					temp_x = frame
					# print(np.array(temp_x).shape)
				elif temp_x.shape[0] < 66*4:
					temp_x = np.concatenate((temp_x, frame), axis=0)
				elif temp_x.shape[0] == 66*4:
					temp_x = np.concatenate((temp_x, frame), axis=0)
					# print(np.array(temp_x).shape)
					X.append([temp_x])
					Y.append([final_data])
				else:
					temp_x = temp_x[66:,::,::]
					temp_x = np.concatenate((temp_x, frame), axis=0)
					# print(np.array(temp_x).shape)
					X.append([temp_x])
					Y.append([final_data])

			X = X[50:]
			Y = Y[50:]
			# print(np.array(X).shape)
			X = np.array(X).reshape([-1, params.network_height, params.img_width, params.img_channels])
			Y = np.array(Y).reshape([-1,1])
			
			val_x.extend(X)
			val_y.extend(Y)
	
	return val_x, val_y

		
	# finish making val data

	
