from model import cnn
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from etaprogress.progress import ProgressBar as pgb


# read files to create mappings
# train dir name to class idx
# val image name to class idx

file = open('class_mapping.txt', 'rb')
class_mapping_list = file.read().split()
class_map = {}
for i in range(0, 400, 2):	class_map[class_mapping_list[i]]=class_mapping_list[i+1]
file.close()

file = open('tiny-imagenet-200/val/val_annotations.txt', 'rb')
annotations_list = file.read().split()
val_annotations = {}
for i in range(0, 60000, 6): val_annotations[annotations_list[i]]=int(class_map[annotations_list[i+1]])
file.close()


# one hot encoding
# returns array of shape (1, dim)

def one_hot(i, dim=180):
	arr = np.zeros((1, dim))
	arr[0][int(i)] = 1
	return arr


# read images from train directory
# store data and labels

train_dir = os.listdir('tiny-imagenet-200/train')
train_data = ()
train_labels = ()
print '\n'
for i in range(len(train_dir)):
	print '\033[Floaded train data:', i*100./len(train_dir), '%'
	im= os.listdir('tiny-imagenet-200/train/'+train_dir[i]+'/images')
	for j in range(len(im)):
		img = Image.open('tiny-imagenet-200/train/'+train_dir[i]+'/images/'+im[j]).convert('RGB')
		img = np.array(img, dtype='float32')/256.
		img = img.reshape(1, 64, 64, 3)
		train_data +=(img, )
		train_labels+=(one_hot(class_map[train_dir[i]]), )
train_data = np.concatenate(train_data)
train_labels = np.concatenate(train_labels)
print train_data.shape, train_labels.shape


# read images from validation directory
# store data and labels

valid_data = ()
valid_labels = ()
print '\n'
valid_dir = os.listdir('tiny-imagenet-200/val/images')
for j in range(len(valid_dir)):
	print '\033[Floaded valid data:', j*100./len(valid_dir), '%'
	img = Image.open('tiny-imagenet-200/val/images/'+valid_dir[j]).convert('RGB')
	img = np.array(img, dtype='float32')/256.
	img = img.reshape(1, 64, 64, 3)
	class_idx = val_annotations[valid_dir[j]]
	if class_idx >= 180:	continue
	valid_data +=(img, )
	valid_labels+=(one_hot(val_annotations[valid_dir[j]]), )
valid_data = np.concatenate(valid_data)
valid_labels = np.concatenate(valid_labels)
print valid_data.shape, valid_labels.shape, '\n'


# training begins here
batch_size = 50

input = tf.placeholder('float32', [batch_size, 64, 64, 3])
label = tf.placeholder('uint8', [batch_size, 180])
t_model = cnn(input, label)
optimizer = tf.train.AdamOptimizer().minimize(t_model.loss)

print 'starting training ...'
n_epochs = 30
n_samples = train_data.shape[0]
n_val_samples = valid_data.shape[0]
n_batches = n_samples/batch_size
n_val_batches = n_val_samples/batch_size
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
if not os.path.exists('./params'):	os.makedirs('./params')
best_val_loss = np.inf
if not os.path.exists('./logs/train'):	os.makedirs('./logs/train')
if not os.path.exists('./logs/val'):	os.makedirs('./logs/val')
tf.summary.scalar('loss', t_model.loss)
tf.summary.scalar('accuracy', t_model.accuracy)
merged = tf.summary.merge_all()


with tf.Session() as sess:
	sess.run(init_op)
	if os.path.exists('./params/cnn_model.ckpt.meta'):
		saver.restore(sess, './params/cnn_model.ckpt')
		print 'model restored from previous checkpoint..'
	train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
	val_writer = tf.summary.FileWriter('./logs/val', sess.graph)
	print 'saving logs to ./logs/ ..'

	for epoch in range(1, n_epochs+1):
		print 
		bar = pgb(n_batches, max_width=50)
		for batch_idx in range(n_batches):
			batch_x = train_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
			batch_y = train_labels[batch_idx*batch_size:(batch_idx+1)*batch_size]
			train_batch_loss, train_acc, op, tr_summary = sess.run([t_model.loss, t_model.accuracy, optimizer, merged], feed_dict={input:batch_x, label:batch_y})
			bar.numerator = batch_idx
			print 'epoch:', epoch, bar, 'train loss:', train_batch_loss, 'accuracy:', train_acc, '\033[F'

		val_losses=[]
		val_accs = []
		vbar = pgb(n_val_batches, max_width=50)
		for vb_idx in range(n_val_batches):
			v_batch_x = valid_data[vb_idx*batch_size:(vb_idx+1)*batch_size]
			v_batch_y = valid_labels[vb_idx*batch_size:(vb_idx+1)*batch_size]
			val_loss, v_acc, v_summary = sess.run([t_model.loss, t_model.accuracy, merged], feed_dict={input:v_batch_x, label:v_batch_y})
			vbar.numerator = vb_idx
			print 'epoch:', epoch, vbar, 'val loss:', val_loss, 'accuracy:', v_acc, '\033[F'
			val_losses.append(val_loss)
			val_accs.append(v_acc)

		cur_val_loss = np.mean(val_losses)
		cur_val_acc = np.mean(val_accs)
		print '\nepoch:', epoch, 'mean validation loss:', cur_val_loss, '& mean validation acc:', cur_val_acc
		if (cur_val_loss < best_val_loss):
			print 'valdation loss improved!'
			save_path = saver.save(sess, "./params/cnn_model.ckpt")
			print("Best model saved in file: %s" % save_path)
			best_val_loss = cur_val_loss
