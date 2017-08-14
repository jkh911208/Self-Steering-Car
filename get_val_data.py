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

			for data in loaded_data:
				tmp = cv2.flip(data[0],0)
				tmp = cv2.flip(tmp,1)
				if params.img_channels != 3:
					tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
				data[0] = (tmp[70:-5,::]).reshape(params.img_height, params.img_width, params.img_channels)

				# change the can data (HEX) to numerical data
				tmp = data[1]
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
				data[1] = final_data

			loaded_data = loaded_data[50:]
			tmp_x = np.array([i[0] for i in loaded_data]).reshape([-1, params.img_height, params.img_width, params.img_channels])
			tmp_y = np.array([i[1] for i in loaded_data]).reshape([-1,1])
			
			val_x.extend(tmp_x)
			val_y.extend(tmp_y)
	
	return val_x, val_y

		
	# finish making val data

	
