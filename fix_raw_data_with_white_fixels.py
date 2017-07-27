import os
import cv2
import numpy as np
import time

'''
original data
width : 256
height : 141
'''

file_list = os.listdir('/raw_data')
#print(data_list)

for file in file_list:
	if file.endswith('.npy'):
		print(file)
		file_location = "/raw_data/" + file
		loaded_file = np.load(file_location)

		fixed_file = loaded_file[45:]
		file_name ="/raw_data/" + "fixed_" + str(file)
		np.save(file_name, fixed_file)
'''
data after resize
width : 256
height : 66
'''


