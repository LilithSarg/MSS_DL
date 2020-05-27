from .BaseNN import *   
from tensorflow.contrib import rnn

class DNN(BaseNN):
  
    def network_lstm(self, X):
        input = tf.unstack(X, self.sequence_length, 1)
        lstm = tf.contrib.rnn.BasicLSTMCell(1000, forget_bias = True)
        outs, _ = tf.contrib.rnn.static_rnn(lstm, input, dtype = "float32")
        outs_ = tf.stack(outs, axis = 0)
        pred = tf.layers.dense(outs_, units = self.fft_length, activation = tf.nn.relu)
        pred_ = tf.reshape(pred, [-1, int(self.sequence_length), int(self.fft_length)])
        return pred_

    def network_d1(self, X):
        pred1 = tf.layers.dense(X, units = self.fft_length, activation = tf.nn.relu)
        return pred1

    def network_d3(self, X):
        pred1 = tf.layers.dense(X, units = self.fft_length, activation = tf.nn.relu)
        pred2 = tf.layers.dense(pred1, units = self.fft_length, activation = tf.nn.relu)
        pred3 = tf.layers.dense(pred2, units = self.fft_length, activation = tf.nn.relu)
        return pred3

    def network_d5(self, X):
        pred1 = tf.layers.dense(X, units = self.fft_length, activation = tf.nn.relu)
        pred2 = tf.layers.dense(pred1, units = self.fft_length, activation = tf.nn.relu)
        pred3 = tf.layers.dense(pred2, units = self.fft_length, activation = tf.nn.relu)
        pred4 = tf.layers.dense(pred3, units = self.fft_length, activation = tf.nn.relu)
        pred5 = tf.layers.dense(pred4, units = self.fft_length, activation = tf.nn.relu)
        return pred5


    def _rnn_placeholders(self, state, c_name):
        c = state
        c = tf.placeholder_with_default(c, c.shape, c_name)
        # h = tf.placeholder_with_default(h, h.shape, h_name)
        # return tf.contrib.rnn.LSTMStateTuple(c, h)
        return c

    def network_bgru(self, batch_x, seq_len):
        self.num_units = 1500
        self.batch_x = batch_x
        # self.n_layers = 2
        with tf.variable_scope("network"):
            gru_layer_fw = tf.contrib.rnn.GRUCell(self.num_units)
            gru_layer_bw = tf.contrib.rnn.GRUCell(self.num_units)
                #gru_layer = tf.contrib.rnn.GRUCell(self.num_units)
                # multi_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.num_units, forget_bias=1.0) for _ in range(self.n_layers)]) 
            initial_state_fw = self._rnn_placeholders(gru_layer_fw.zero_state(tf.shape(batch_x)[0], tf.float32), "c_state_fw")
            initial_state_bw = self._rnn_placeholders(gru_layer_bw.zero_state(tf.shape(batch_x)[0], tf.float32), "c_state_bw")
            outputs, current_state = tf.nn.bidirectional_dynamic_rnn(gru_layer_fw, gru_layer_bw, batch_x, sequence_length=seq_len, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, dtype="float32")
            outputs = tf.concat(outputs, 2)
            dense1 = tf.layers.dense(inputs=outputs, units=3000, activation=tf.nn.relu, name="dense1")
            lin1 = tf.layers.dense(inputs=outputs, units=3000, activation=None, name="lin1")
            sum1 = dense1 + lin1
            dense2 = tf.layers.dense(inputs=sum1, units=3000, activation=tf.nn.relu, name="dense2")
            lin2 = tf.layers.dense(inputs=sum1, units=3000, activation=None, name="lin2")
            sum2 = dense2 + lin2
            prediction = tf.layers.dense(inputs=sum2, units=self.fft_length, activation=tf.sigmoid, name="last_dense")
        return prediction

    def metrics(self, Y, Y_pred):
        cost = tf.reduce_mean(tf.square(Y_pred - Y))
        tf.summary.scalar('cost_funtion', cost)
        return cost