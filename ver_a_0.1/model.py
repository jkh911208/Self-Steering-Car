import tensorflow as tf
import params

print_layer = True

x = tf.placeholder(tf.float32, shape=[None, params.network_height, params.img_width, params.img_channels])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, params.network_height, params.img_width, params.img_channels])

network = tf.layers.batch_normalization(x_image, name="norm_0")
if print_layer:
	print(network)

# Conv Layer # 1
network = tf.layers.conv2d(network, filters=96, kernel_size = (11,11), strides=(4,4), 
	padding='same', activation=tf.nn.relu, use_bias=True,name="conv_1")
if print_layer:
	print(network)
network = tf.layers.max_pooling2d(network, pool_size=(3,3), strides=2, padding='same', 
	name="m_pooling_1")
if print_layer:
	print(network)
network = tf.layers.batch_normalization(network, name="norm_1")
if print_layer:
	print(network)

# Conv Layer #2

network = tf.layers.conv2d(network, filters=256, kernel_size = (5,5), strides=(1,1), 
	padding='same', activation=tf.nn.relu, use_bias=True,name="conv_2")
if print_layer:
	print(network)
network = tf.layers.max_pooling2d(network, pool_size=(3,3), strides=2, padding='same', 
	name="m_pooling_2")
if print_layer:
	print(network)
network = tf.layers.batch_normalization(network, name="norm_2")
if print_layer:
	print(network)

# Conv Layer #3

network = tf.layers.conv2d(network, filters=384, kernel_size = (3,3), strides=(1,1), 
	padding='same', activation=tf.nn.relu, use_bias=True,name="conv_3_1")
if print_layer:
	print(network)
network = tf.layers.conv2d(network, filters=256, kernel_size = (3,3), strides=(1,1), 
	padding='same', activation=tf.nn.relu, use_bias=True,name="conv_3_2")
if print_layer:
	print(network)
network = tf.layers.conv2d(network, filters=32, kernel_size = (3,3), strides=(1,1), 
	padding='same', activation=tf.nn.relu, use_bias=True,name="conv_3_3")
if print_layer:
	print(network)
network = tf.layers.max_pooling2d(network, pool_size=(3,3), strides=2, padding='same', 
	name="m_pooling_3")
if print_layer:
	print(network)
network = tf.layers.batch_normalization(network, name="norm_3")
if print_layer:
	print(network)

# Flatten
network = tf.reshape(network, [-1, 2816], name="flatten")
if print_layer:
	print(network)

# Dense Layer 1

network = tf.layers.dense(network, 4096, activation=tf.nn.relu, name="dense_1")
if print_layer:
	print(network)
network = tf.layers.dropout(network, rate=keep_prob, name="dropout_1")
if print_layer:
	print(network)

# Dense Layer 2

network = tf.layers.dense(network, 4096, activation=tf.nn.relu, name="dense_2")
if print_layer:
	print(network)
network = tf.layers.dropout(network, rate=keep_prob, name="dropout_2")
if print_layer:
	print(network)

# # Dense Layer 3

# network = tf.layers.dense(network, 1000, activation=tf.nn.relu, name="dense_3")
# if print_layer:
# 	print(network)
# network = tf.layers.dropout(network, rate=keep_prob, name="dropout_3")
# if print_layer:
# 	print(network)

# output layer

# activation=None => linear
y = tf.layers.dense(network, 1, activation=None, name="output")
if print_layer:
	print(y)