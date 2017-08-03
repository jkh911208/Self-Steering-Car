#!/usr/bin/env python 

import os
import numpy as np

file_list = os.listdir('/raw_data')
for file_name in file_list:
	if file_name.endswith('.npy'):
		file_location = "/raw_data/" + file_name
		loaded_data = np.load(file_location)

		for data in loaded_data:
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

			# put the int and the decimal together
			final_data = int_data + int_decimal

			if final_data > 20 or final_data < -20:
				print("Bad data : ", file_name)
				break



