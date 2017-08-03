#!/usr/bin/env python 
from __future__ import division

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
#     tf.summary.scalar("loss", loss)

# merge all summaries into a single op
# if write_summary:
#     merged_summary_op = tf.summary.merge_all()

# saver = tf.train.Saver()
time_start = time.time()

# op to write logs to Tensorboard
# if write_summary:
#     summary_writer = tf.summary.FileWriter(params.save_dir, graph=tf.get_default_graph())

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

# print(mnist.train.images.shape)

# train_X = tf.reshape(mnist.train.images, [-1,28,28,1])
# train_Y = tf.reshape(mnist.train.labels, [-1,1])

val_X = tf.reshape(mnist.test.images, [-1,28,28,1])
val_Y = tf.reshape(mnist.test.labels, [-1,1])

# print(train_X.shape, train_Y.shape,val_X.shape,val_Y.shape)

for i in range(params.epoch):   
    print ("start train step {} of {}".format(i, params.epoch))
    batch_X, batch_Y = mnist.train.next_batch(100)
    batch_X = batch_X.reshape([-1,28,28,1])
    batch_Y = batch_Y.reshape([-1,1])

    train_step.run(feed_dict={model.x: batch_X, model.y_: batch_Y, model.keep_prob: 0.8})

    # write logs at every iteration
    # if write_summary:
    #     summary = merged_summary_op.eval(feed_dict={model.x: train_X, model.y_: train_Y, model.keep_prob: 1.0})
    #     summary_writer.add_summary(summary, i)

    t_loss = loss.eval(feed_dict={model.x: batch_X, model.y_: batch_Y, model.keep_prob: 1.0})
    v_loss = loss.eval(feed_dict={model.x: val_X, model.y_: val_Y, model.keep_prob: 1.0})
    print ("epoch {} of {}, train loss {}, val loss {}".format(i, params.epoch,t_loss, v_loss))

    # if (i+1) % 100 == 0:
    #     if not os.path.exists(params.save_dir):
    #         os.makedirs(params.save_dir)
    #     checkpoint_path = os.path.join(params.save_dir, "model.ckpt")
    #     filename = saver.save(sess, checkpoint_path)

    #     time_passed = cm.pretty_running_time(time_start)
    #     time_left = cm.pretty_time_left(time_start, i, params.training_steps)
    #     print 'Model saved. Time passed: {}. Time left: {}'.format(time_passed, time_left) 
        
