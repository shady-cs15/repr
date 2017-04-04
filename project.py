from model import project_cnn
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np


f = Image.open('rgb.jpg').convert('RGB')
f = np.array(f, dtype='float32')/256.
f = f.transpose(2, 0, 1)
f = f.reshape(1, 3, 64, 64)


p_model = project_cnn(verbose=True)
p_model.load_weights('cnn.hdf5', by_name=True)
print p_model.predict(f).shape