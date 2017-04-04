from model import train_cnn
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
from keras.callbacks import ModelCheckpoint

f = Image.open('rgb.jpg').convert('RGB')
f = np.array(f, dtype='float32')/256.
f = f.transpose(2, 0, 1)
f = f.reshape(1, 3, 64, 64)
f = np.concatenate([f, f])
y = np.zeros((2, 180))
y[0][0] = 1
t_model = train_cnn()
print t_model.predict(f).shape

weights_file='cnn.hdf5'
t_model.compile(loss='binary_crossentropy', optimizer='adam')
checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
t_model.fit(f, y, validation_split=0.5, nb_epoch=100, batch_size=1, callbacks=callbacks_list, verbose=1)