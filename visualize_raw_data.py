import os
import cv2
import numpy as np
import time

'''
original data
width : 256
height : 141
'''

file_list = os.listdir('/raw_data/bad_data')
#print(data_list)

for file in file_list:
	if file.endswith('.npy'):
		print(file)
		file_location = "/raw_data/bad_data/" + file
		loaded_data = np.load(file_location)
		number = 0

		for data in loaded_data:
			tmp = cv2.flip(data[0],0)
			tmp = cv2.flip(tmp,1)
			frame = tmp[70:-5,::]
			cv2.imshow('frame', frame)
			print(frame.shape, data[1], number)
			number += 1
		
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

'''
data after resize
width : 256
height : 66
'''


