import numpy as np
import os
import cv2
import time


from networks import googLeNet

WIDTH = 256
HEIGHT = 66
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'Self-Steering-Car-{}-{}-{}-epochs.model'.format('googLeNet', LR , EPOCHS)

model = googLeNet(HEIGHT,WIDTH, LR)

# start training
file_list = os.listdir('/raw_data')
for file in file_list:
	if file.endswith('.npy'):
		print("Start training on file : " + file + ".npy")
		file_location = "/raw_data/" + file
		loaded_data = np.load(file_location)

		X = []
		Y = []

		temp_x = []

		for data in loaded_data:
			frame = (data[0][70:-5,::]).reshape(66,256,1)
			
			if len(temp_x) == 0:
				temp_x = frame
			elif temp_x.shape[2] < 4:
				temp_x = np.concatenate((temp_x, frame), axis=2)
			elif temp_x.shape[2] == 4:
				temp_x = np.concatenate((temp_x, frame), axis=2)
				X.append([temp_x])
				Y.append([data[1]])
			else:
				temp_x = temp_x[::,::,-4:]
				#print(temp_x.shape)
				temp_x = np.concatenate((temp_x, frame), axis=2)
				#print(temp_x.shape)
				X.append([temp_x])
				Y.append([data[1]])
				#print(len(X))
				#print(len(Y))


			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		X = np.array(X).reshape(-1,66,256,5)
		#print(X.shape)
		Y = np.array(Y).reshape(-1,1)
		#print(Y.shape)
		
		model.fit(X,Y,run_id=MODEL_NAME, show_metric=True,shuffle=True,snapshot_step=500)

		model.save(MODEL_NAME)
