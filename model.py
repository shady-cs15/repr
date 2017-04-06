import numpy as np
import theano

import tensorflow as tf

class cnn():
	def __init__(self, input, labels, batch_size=50, is_training=False):
		self.input_layer = tf.reshape(input, shape=[batch_size, 64, 64, 3])
		self.conv1 = tf.layers.conv2d(
      		inputs=self.input_layer,
      		filters=16,
      		kernel_size=[3, 3],
      		padding="same",
      		activation=tf.nn.relu)
		self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)
		
		self.conv2 = tf.layers.conv2d(
      		inputs=self.pool1,
      		filters=32,
      		kernel_size=[3, 3],
      		padding="same",
      		activation=tf.nn.relu)
		self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2)
		
		self.conv3 = tf.layers.conv2d(
      		inputs=self.pool2,
      		filters=64,
      		kernel_size=[3, 3],
      		padding="same",
      		activation=tf.nn.relu)
		self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3, pool_size=[2, 2], strides=2)
		
		self.conv4 = tf.layers.conv2d(
      		inputs=self.pool3,
      		filters=128,
      		kernel_size=[3, 3],
      		padding="same",
      		activation=tf.nn.relu)
		self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4, pool_size=[2, 2], strides=2)
		
		self.conv5 = tf.layers.conv2d(
      		inputs=self.pool4,
      		filters=256,
      		kernel_size=[3, 3],
      		padding="same",
      		activation=tf.nn.relu)
		self.pool5 = tf.layers.max_pooling2d(inputs=self.conv5, pool_size=[2, 2], strides=2)
		
		self.conv6 = tf.layers.conv2d(
      		inputs=self.pool5,
      		filters=512,
      		kernel_size=[3, 3],
      		padding="same",
      		activation=tf.nn.relu)
		self.pool6 = tf.layers.max_pooling2d(inputs=self.conv6, pool_size=[2, 2], strides=2)
		
		self.pool6_flat = tf.reshape(self.pool6, [batch_size, 512])
  		self.dense1 = tf.layers.dense(inputs=self.pool6_flat, units=300, activation=tf.nn.tanh)
  		self.dense2 = tf.layers.dense(inputs=self.dense1, units=180, activation=tf.nn.tanh)
  		
  		self.pred = tf.nn.softmax(self.dense2)
  		self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
  		self.loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=self.dense2)


