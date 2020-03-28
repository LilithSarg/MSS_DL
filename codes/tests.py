import numpy as np
import os
# import tensorflow as tf
import glob
from random import randrange

print(os.getcwd())
os.chdir('../../MUSDBHQ-18_sample/test/Arise - Run Run Run/mixture_prep')
print(os.getcwd())
for i in os.listdir():
	f = np.load(i)
	print(i, '-->', f.shape)


# X = tf.placeholder('float', [100, 199, 802], name = 'X')
# Y = tf.placeholder('float', [100, 199, 802], name = 'Y')
# input = tf.unstack(value = X, num = 199, axis = 1)
# lstm = tf.contrib.rnn.BasicLSTMCell(100, forget_bias = True)
# outs, _ = tf.contrib.rnn.static_rnn(lstm, input, dtype = "float32")
# outs_ = tf.stack(outs, axis = 0)
# pred = tf.layers.dense(outs_, units = 802, activation = tf.nn.sigmoid)
# # print(tf.shape(input)) # --> Tensor("Shape:0", shape=(3,), dtype=int32)
# # print(len(input))      # --> 199
# # print('OUTS TYPE -->', type(outs)) # --> OUTS TYPE --> <class 'list'>
# # print('OUTS TYPE -->', outs[0].shape)
# # print('OUTS_NEW TYPE --> ', type(outs_) ) # --> tensor
# print('PRED TYPE -->', type(pred))
# print('PRED SHAPE -->', pred.shape)

# print('CORRECT SHAPE -->', Y.shape)
# print('CORRECT RESHAPEd -->', tf.reshape(Y, [199, 100, 802]))
# Y_reshape = tf.reshape(Y, [199, 100, 802])
# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_reshape, 1))
# cost = tf.reduce_mean(tf.square(pred - Y_reshape))
# accuracy = tf.reduce_mean(tf.cast(pred, 'float'))

# train_images_dir = '/home/student/Desktop/Lilith_dont_delete_anything/Deep Learning/data_sample/val_art'
# test_paths_songs = glob.glob(os.path.join(train_images_dir, '*\\'))
# test_paths_seconds = glob.glob(os.path.join(train_images_dir, '*\\*\\*'))
# for path in test_paths_seconds:
# 	print(path)
    # seconds = len(os.listdir(path))
    # sec = randrange(seconds)
    # print(path + '\\mixture_prep\\' + '{}_mixture_MP.npy'.format(sec))  #'matrix'_{}'.format(sec)
    # label  = np.load(path + '\\vocals_prep\\' + '{}_vocals_MP.npy'.format(sec))
	# print(path)

# print('ksabk-->', round(len(test_paths_seconds) / 100))
# print('TEST_PATHS LEN -->', len(test_paths_seconds))

# test_images_dir = '/home/student/Desktop/Lilith_dont_delete_anything/Deep Learning/data_sample/val_art'
# train_paths = glob.glob(os.path.join(train_images_dir, '*\\*\\*'), recursive=True)

# def load_sec_mpa(path): # this song_path 
#     matrix = np.load(path)
#     label  = np.load(path) 
#     return matrix, label

# def batch_data_loader(batch_size, file_paths, index):
#     matrix = []
#     labels = []   
#     for image in file_paths[index*batch_size : (index+1)*batch_size]:
#         matrix.append(load_sec_mpa(image)[0])
#         labels.append(load_sec_mpa(image)[1])
#     return matrix, labels

# def test_data_loader(index):
#     return batch_data_loader(test_batch_size, test_paths, index)

# def test_model():
#     minibatch_full_test = round(len(test_paths) / test_batch_size)
#     x_test=[]
#     y_test=[]
#     for i in range(minibatch_full_test):
#         x_test_, y_test_= data_loader.test_data_loader(i)
#         x_test = x_test + x_test_
#         y_test = y_test + y_test_
#     print('X_TEST SHAPE IS -->', x_test)