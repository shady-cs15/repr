from model_utils import add_conv_pool_layer, add_deconv_unpool_layer
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten

def create_train_model():
	model = Sequential()
	model = add_conv_pool_layer(model, 16, (3, 64, 64))
	model = add_conv_pool_layer(model, 32)
	model = add_conv_pool_layer(model, 64)
	model = add_conv_pool_layer(model, 128)
	model = add_conv_pool_layer(model, 256)
	model = add_deconv_unpool_layer(model, 128)
	model = add_deconv_unpool_layer(model, 64)
	model = add_deconv_unpool_layer(model, 32)
	model = add_deconv_unpool_layer(model, 16)
	model = add_deconv_unpool_layer(model, 3, act=False)
	print 'train model created with following summary..'
	print model.summary()
	return model

def create_rep_model(wts):
	assert len(wts)==5
	model = Sequential()
	model = add_conv_pool_layer(model, 16, (3, 64, 64), wt=wts[0])
	model = add_conv_pool_layer(model, 32, wt=wts[1])
	model = add_conv_pool_layer(model, 64, wt=wts[2])
	model = add_conv_pool_layer(model, 128, wt=wts[3])
	model = add_conv_pool_layer(model, 256, wt=wts[4])
	model.add(Flatten())
	print 'rep model created with following summary..'
	print model.summary()
	return model
