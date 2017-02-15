from model_utils import add_conv_pool_layer, add_deconv_unpool_layer
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np

from keras.models import Sequential

f = Image.open('rgb.jpg').convert('RGB')
f = np.array(f, dtype='float32')/256.
f = f.transpose(2, 0, 1)

f = f.reshape(1, 3, 480, 640)
#print f.shape

model = Sequential()
model = add_conv_pool_layer(model, 16, (3, 480, 640))
model = add_deconv_unpool_layer(model, 3, (None, 3, 480, 640))
print model.summary()

y = model.predict(f)
print y[0][0].shape
plt.gray()
plt.figure(1)
plt.imshow(y[0][0])
plt.show()
