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

# write_summary = params.write_summary

sess = tf.InteractiveSession()

loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# create a summary to monitor cost tensor
# if write_summary:
# 	tf.summary.scalar("loss", loss)

# merge all summaries into a single op
# if write_summary:
# 	merged_summary_op = tf.summary.merge_all()

# saver = tf.train.Saver()
time_start = time.time()

# op to write logs to Tensorboard
# if write_summary:
# 	summary_writer = tf.summary.FileWriter(params.save_dir, graph=tf.get_default_graph())


# get the val data
val_X, val_Y = val_data()
# finish get val data

file_list = os.listdir('/raw_data')
for i in range(params.epoch):   
	# prepare data for training
	for file in file_list:
		if file.endswith('.npy'):
			print("Start process on file : " , file)
			train_X, train_Y = train_data(file)
		    	#finishing getting the training data
		
			# start the train on the data
			batch_iteration = int(train_X.shape[0] / params.batch) + 1
			for iteration in range(batch_iteration):
				batch_X = train_X[iteration*params.batch:(iteration+1)*params.batch]
				batch_Y = train_Y[iteration*params.batch:(iteration+1)*params.batch]

				train_step.run(feed_dict={model.x: batch_X, model.y_: batch_Y, model.keep_prob: 0.8})

			# write logs at every iteration
			# if write_summary:
			#     summary = merged_summary_op.eval(feed_dict={model.x: train_X, model.y_: train_Y, model.keep_prob: 1.0})
			#     summary_writer.add_summary(summary, i)

				t_loss = loss.eval(feed_dict={model.x: batch_X, model.y_: batch_Y, model.keep_prob: 1.0})
				v_loss = loss.eval(feed_dict={model.x: val_X, model.y_: val_Y, model.keep_prob: 1.0})
				print ("epoch {} of {}, batch {} of {}, train loss {}, val loss {}".format(i, params.epoch,iteration,batch_iteration,t_loss, v_loss))

			# if (i+1) % 10 == 0:
			# 	if not os.path.exists(params.save_dir):
 		# 			os.makedirs(params.save_dir)
			# 	checkpoint_path = os.path.join(params.save_dir, "model.ckpt")
			# 	filename = saver.save(sess, checkpoint_path)



			    
