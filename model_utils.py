


from keras.layers import Activation
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Deconvolution2D
from keras.regularizers import l2

#i_shape = (3, 64, 64) @ start
#n_filters = 16 @ start
def add_conv_pool_layer(model, n_filters, i_shape=None, k_size=(3, 3), p_shape=(2, 2), wt=None):
	if i_shape is not None:
		model.add(ZeroPadding2D(input_shape=i_shape))
	else:
		model.add(ZeroPadding2D())
	if wt is None:
		model.add(Convolution2D(n_filters, k_size[0], k_size[1], W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
	else:
		model.add(Convolution2D(n_filters, k_size[0], k_size[1], W_regularizer=l2(0.01), b_regularizer=l2(0.01), weights=wt))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=p_shape))
	return model

def add_deconv_unpool_layer(model, n_filters, k_size=(3, 3), p_shape=(2, 2), act= True):
	model.add(UpSampling2D(size=p_shape))
	model.add(ZeroPadding2D())
	model.add(Convolution2D(n_filters, k_size[0], k_size[1], W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
	if act==True:
		model.add(Activation('relu'))
	return model
