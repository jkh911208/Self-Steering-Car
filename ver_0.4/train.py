#!/usr/bin/env python 

import os
import tensorflow as tf
import model
import params
import time
import cv2
import numpy as np

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


# make the val data
file_list = os.listdir('/raw_data/eval_data')
for file in file_list:
    if file.endswith('.npy'):
        file_location = "/raw_data/eval_data/" + file
        loaded_data = np.load(file_location)

        for data in loaded_data:
            tmp = cv2.flip(data[0],0)
            tmp = cv2.flip(tmp,1)
            if params.img_channels == 1:
                tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            data[0] = (tmp[70:-5,::]).reshape(params.img_height, params.img_width, params.img_channels)

            # change the can data (HEX) to numerical data
            tmp = data[1]
            hex_data = tmp[-23:-21] + tmp[-20:-18]
            hex_decimal = tmp[-3:-1]
            int_data = int(hex_data, 16)
            int_decimal = int(hex_decimal, 16)
            
            # if the steering wheel angle in in right to the center
            if(int_data > 550):
                int_data = int_data - 4096
                int_decimal = 256 - int_decimal 
            
            # put the int and the decimal together
            num_in_string = str(int_data) + "." + str(int_decimal)
            final_data = float(num_in_string)
            data[1] = final_data
        
        loaded_data = loaded_data[30:]
        val_X = np.array([i[0] for i in loaded_data]).reshape([-1, params.img_height, params.img_width, params.img_channels])
        val_Y = np.array([i[1] for i in loaded_data]).reshape([-1,1])
# finish making val data

file_list = os.listdir('/raw_data')
for i in range(params.epoch):   
	# prepare data for training
	for file in file_list:
		if file.endswith('.npy'):
			print("Start training on file : " , file)
			file_location = "/raw_data/" + file
			loaded_data = np.load(file_location)

			# change the saved data into the form that both human and NN can understand
			for data in loaded_data:
				# flip the frame because the webcam is mounted upside down on the front windsheild, and cut out the sky portion of the image the size became 256*66
				tmp = cv2.flip(data[0],0)
				tmp = cv2.flip(tmp,1)
				if params.img_channels == 1:
					tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
				data[0] = (tmp[70:-5,::]).reshape(params.img_height,params.img_width,params.img_channels)

				# change the can data (HEX) to numerical data
				tmp = data[1]
				hex_data = tmp[-23:-21] + tmp[-20:-18]
				hex_decimal = tmp[-3:-1]
				int_data = int(hex_data, 16)
				int_decimal = int(hex_decimal, 16)

				# if the steering wheel angle in in right to the center
				if(int_data > 550):
					int_data = int_data - 4096
					int_decimal = 256 - int_decimal 

				# put the int and the decimal together
				num_in_string = str(int_data) + "." + str(int_decimal)
				final_data = float(num_in_string)
				data[1] = final_data
				#print(final_data)
			#did cut out the first few frames because the webcam need time to adjust the exposure
			# but seems like still the white out frames exist

			loaded_data = loaded_data[30:]
			train_X = np.array([i[0] for i in loaded_data]).reshape([-1, params.img_height,params.img_width,params.img_channels])
			train_Y = np.array([i[1] for i in loaded_data]).reshape([-1,1])

		    #finishing making the training data

			print ("start train step {} of {}".format(i, params.epoch))
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



			    
