import numpy as np
import os
import cv2
import time

# Load MNIST data set
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=False)
X = X.reshape([-1, 28, 28, 1])
Y = Y.reshape([-1,1])
testX = testX.reshape([-1, 28, 28, 1])
testY = testY.reshape([-1,1])

#print(Y)

WIDTH = 28
HEIGHT = 28
CHANNEL = 1

from networks import googLeNet, AlexNet

LR = 1e-2
EPOCHS = 10
MODEL_NAME = 'MNIST-{}-{}-{}-epochs.model'.format('AlexNet', LR , EPOCHS)

model = AlexNet(HEIGHT,WIDTH,CHANNEL, LR)

		# train the network
model.fit(X, Y, n_epoch=EPOCHS, run_id=MODEL_NAME, show_metric=True, validation_set=(testX,testY))

model.save(MODEL_NAME)


