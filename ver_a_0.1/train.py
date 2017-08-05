#!/usr/bin/env python 

import os
import tensorflow as tf
import model
import params
import time
import cv2
import numpy as np

from get_val_data import val_data
from get_train_data import train_data

sess = tf.InteractiveSession()

loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# get the val data
val_X, val_Y = val_data()
val_X = val_X[600:]
val_Y = val_Y[600:]
# print(np.array(val_Y).shape)
# finish get val data

file_list = os.listdir('/raw_data')
for i in range(params.epoch):   
	# prepare data for training
	for file in file_list:
		if file.endswith('.npy'):
			train_X, train_Y = train_data(file)
		    	#finishing getting the training data
			print("Start train on file : " , file)
			# start the train on the data
			batch_iteration = int(train_X.shape[0] / params.batch)
			# print(train_X.shape[0])
			if train_X.shape[0] / params.batch > batch_iteration:
				batch_iteration = batch_iteration + 1

			for iteration in range(batch_iteration):
				# print(iteration)
				batch_X = train_X[iteration*params.batch:(iteration+1)*params.batch]
				batch_Y = train_Y[iteration*params.batch:(iteration+1)*params.batch]
				# print(np.array(batch_X).shape)
				# print(np.array(batch_Y).shape)


				train_step.run(feed_dict={model.x: batch_X, model.y_: batch_Y, model.keep_prob: 0.5})
				# print("train done for batch")

				t_loss = loss.eval(feed_dict={model.x: batch_X, model.y_: batch_Y, model.keep_prob: 1.0})
				v_loss = loss.eval(feed_dict={model.x: val_X, model.y_: val_Y, model.keep_prob: 1.0})
				print ("epoch {} of {}, batch {} of {}, batch loss {}, val loss {}".format(i, params.epoch,iteration,batch_iteration,t_loss, v_loss))


			    
