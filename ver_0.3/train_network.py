import numpy as np
import os
import cv2

from networks import googLeNet

WIDTH = 256
HEIGHT = 66
LR = 1e-2
EPOCHS = 10
MODEL_NAME = 'Self-Steering-Car-{}-{}-{}-epochs.model'.format('googLeNet', LR , EPOCHS)

model = googLeNet(HEIGHT,WIDTH, LR)

# start training
file_list = os.listdir('/raw_data')
for file in file_list:
	if file.endswith('.npy'):
		print("Start training on file : " + file)
		file_location = "/raw_data/" + file
		loaded_data = np.load(file_location)

		for data in loaded_data:
			# flip the frame because the webcam is mounted upside down on the front windsheild, and cut out the sky portion of the image the size became 256*66
			tmp = cv2.flip(data[0],0)
			tmp = cv2.flip(tmp,1)
			data[0] = (tmp[70:-5,::]).reshape(66,256,1)

			# change the can data to numerical data

		train = loaded_data[:-100]
		test = loaded_data[-100:]

		train_X = np.array([i[0] for i in train]).reshape(-1, HEIGHT,WIDTH,1)
		train_Y = np.array([i[1] for i in train]).reshape(-1,1)

		val_X = np.array([i[0] for i in test]).reshape(-1, HEIGHT,WIDTH,1)
		val_Y = np.array([i[1] for i in test]).reshape(-1,1)


		
		model.fit(train_X, train_Y, n_epoch=EPOCHS, run_id=MODEL_NAME, show_metric=True, validation_set=(val_X,val_Y))

		model.save(MODEL_NAME)
