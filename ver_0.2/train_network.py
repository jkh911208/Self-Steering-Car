import numpy as np
import os
import cv2
import time
from random import shuffle

from networks import AlexNet

WIDTH = 256
HEIGHT = 66
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'Self-Steering-Car-{}-{}-{}-epochs.model'.format('AlexNet', LR , EPOCHS)

model = AlexNet(HEIGHT,WIDTH, LR)

# start training
file_list = os.listdir('/raw_data')
for file in file_list:
	if file.endswith('.npy'):
		print("Start training on file : " + file)
		file_location = "/raw_data/" + file
		loaded_data = np.load(file_location)

		train = loaded_data[:-100]
		test = loaded_data[-100:]

		X = np.array([i[0][70:-5,::] for i in train]).reshape(-1, HEIGHT,WIDTH,1)
		Y = np.array([i[1] for i in train]).reshape(-1,1)

		val_X = np.array([i[0][70:-5,::] for i in test]).reshape(-1, HEIGHT,WIDTH,1)
		val_Y = np.array([i[1] for i in test]).reshape(-1,1)

		model.fit(X, Y, n_epoch=EPOCHS, run_id=MODEL_NAME, show_metric=True, validation_set=(val_X,val_Y))
		model.save(MODEL_NAME)
