from .BaseNN import *
from tensorflow.contrib import rnn

class DNN(BaseNN):
  
    def network(self, X):
        input = tf.unstack(value = X, num = self.sequence_length, axis = 1)
        lstm = tf.contrib.rnn.BasicLSTMCell(self.train_batch_size, forget_bias = True)
        outs, _ = tf.contrib.rnn.static_rnn(lstm, input, dtype = "float32")
        outs_ = tf.stack(outs, axis = 0)
        pred = tf.layers.dense(outs_, units = self.fft_length, activation = tf.nn.sigmoid)
        return pred
      
    def metrics(self, Y, Y_pred):
        cost = tf.reduce_mean(tf.square(Y_pred - Y))
        tf.summary.scalar('cost_funtion', cost)
        return cost
