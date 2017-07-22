import numpy as np
import os

from networks import googLeNet

WIDTH = 66
HEIGHT = 256
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'Self-Steering-Car-{}-{}-{}-epochs.model'.format('googLeNet', LR , EPOCHS)

model = googLeNet(WIDTH, HEIGHT, LR)

# start training
file_list = os.listdir('/raw_data')
for file in file_list:
	if file.endswith('.npy'):
		print("Start training on file : " + file + ".npy")
		file_location = "/raw_data/" + file
		file = np.load(file_location)

		X = []
		Y = []

		temp_x = []

		for data in file:
			frame = (data[0][70:-5,::]).reshape(66,256,1)

			if len(temp_x) == 0:
				temp_x = frame
				#print(temp_x.shape)
			elif temp_x.shape[2] < 4:
				temp_x = np.concatenate((temp_x, frame), axis=2)
				#print(temp_x.shape)
			else:
				if temp_x.shape[2] == 4:
					temp_x = np.concatenate((temp_x, frame), axis=2)
					#print(temp_x.shape)
					X.append([temp_x])
					Y.append([data[1]])
				else:
					temp_x = temp_x[::,::,-4:]
					#print(temp_x.shape)
					temp_x = np.concatenate((temp_x, frame), axis=2)
					#print(temp_x.shape)
					X.append([temp_x])
					Y.append([data[1]])

		X = np.array(X).reshape(-1,66,256,5)
		#print(X.shape)
		Y = np.array(Y)
		#print(Y.shape)
		
		model.fit(X,Y,run_id=MODEL_NAME)

		model.save(MODEL_NAME)
