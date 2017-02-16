from model import create_train_model, create_rep_model
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
import theano

from keras.models import Sequential

f = Image.open('rgb.jpg').convert('RGB')
f = np.array(f, dtype='float32')/256.
f = f.transpose(2, 0, 1)

f = f.reshape(1, 3, 64, 64)
#print f.shape

t_model= create_train_model()

n_rep_layers = 5
wts = []
for layer in t_model.layers:
	if (len(wts)==n_rep_layers):
		 break
	if len(layer.get_weights())>0:
		wts.append(layer.get_weights())
	else:
		continue
assert n_rep_layers==len(wts)

r_model = create_rep_model(wts)
y = r_model.predict(f)
print y.shape#len(y[0])#[0][0].shape
#plt.gray()
#plt.figure(1)
#plt.imshow(y[0][0])
#plt.show()
