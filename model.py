from model_utils import add_conv_pool_layer
import numpy as np
import theano

from keras.models import Sequential
from keras.layers import Flatten, Dense

def project_cnn(verbose=False):
	model = Sequential()
	model = add_conv_pool_layer(model, 16, (3, 64, 64))
	model = add_conv_pool_layer(model, 32)
	model = add_conv_pool_layer(model, 64)
	model = add_conv_pool_layer(model, 128)
	model = add_conv_pool_layer(model, 256)
	model = add_conv_pool_layer(model, 512)
	model.add(Flatten())
	model.add(Dense(2, input_shape=(512,)))
	if verbose==True:	print model.summary()
	return model

def train_cnn():
	model = project_cnn()
	model.add(Dense(300, input_shape=(2,)))
	model.add(Dense(180, input_shape=(300,), activation='softmax'))
	print model.summary()
	return model