#!/usr/bin/env python 

import os
import tensorflow as tf
import model
import params
import cv2
import numpy as np

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "./model_1")

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
			img = (tmp[70:-5,::]).reshape(params.img_height, params.img_width, params.img_channels)

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

			deg = model.y.eval(feed_dict={model.x: [img], model.keep_prob: 1.0})[0][0]

			print(deg, final_data)


			    
