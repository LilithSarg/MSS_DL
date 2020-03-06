import tensorflow as tf
from data_loader import *
from abc import abstractmethod
import random
import math

class BaseNN:
    def __init__(self, train_spec_dir, val_spec_dir, test_spec_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, sequence_length, fft_length, 
                 learning_rate, base_dir, max_to_keep, model_name):

        self.data_loader = DataLoader(train_spec_dir, val_spec_dir, test_spec_dir, train_batch_size, 
                val_batch_size, test_batch_size, sequence_length, fft_length)

        self.train_paths = glob.glob(os.path.join(train_spec_dir, '*/*/*'), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_spec_dir, '*/*/*'), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_spec_dir, '*/*/*'), recursive=True)
        
        self.sequence_length = sequence_length
        self.fft_length = fft_length 
        self.learning_rate = learning_rate 
        self.train_batch_size = train_batch_size
        self.num_epochs = num_epochs
        self.train_spec_dir = train_spec_dir
        self.val_spec_dir = val_spec_dir
        self.val_batch_size= val_batch_size
        self.test_batch_size = test_batch_size
        self.test_spec_dir =test_spec_dir
        self.base_dir = base_dir
        self.max_to_keep = max_to_keep
        self.model_name = model_name

    def create_network(self):                    # 100                   199                  800
        self.X = tf.placeholder('float', [self.train_batch_size, self.sequence_length, self.fft_length], name = 'X')
        self.Y = tf.placeholder('float', [self.train_batch_size, self.sequence_length, self.fft_length], name = 'Y')
        self.Y_pred = self.network(self.X)
        self.cost = self.metrics(self.Y, self.Y_pred)[0]
        self.accuracy = self.metrics(self.Y, self.Y_pred)[1]
        self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
      
    def initialize_network(self):
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        self.sess = tf.InteractiveSession()
        if os.path.exists(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
            self.sess.run(tf.global_variables_initializer())
        else:
            checkpoint_dir = os.path.join(os.getcwd(), self.base_dir, self.model_name, 'chekpoints')
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))   
      
    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
            print('MODEL TRAIN ---------------> STARTED ')
            minibatch_full = round(len(self.train_paths)/self.train_batch_size)
            minibatch_full_v = round(len(self.val_paths)/self.val_batch_size)
            print('TRAIN MINIBATCHES -->', minibatch_full)

            val_data = []
            print('SPECTOGRAMS APPENDING FOR VAL --------------------> STARTED')
            for spec in range(minibatch_full_v):
                val_data.append(self.data_loader.val_data_loader(spec))
            print('SPECTOGRAMS APPENDING FOR VAL --------------------> ENDED')
            print('EPOCHES -----------------------> STARTED')
            for epoch in range(self.num_epochs):
                print('EPOCH NUMBER -----> ', epoch)
                random.shuffle(self.data_loader.train_paths)
                minibatch_spec = []
                for k_th_batch in range(minibatch_full):
                    minibatch_spec.append(self.data_loader.train_data_loader(k_th_batch))

                epoch_cost = 0
                epoch_accuracy = 0
                print('SPECTOGRAMS APPENDING FOR MINIBATCH --------------------> STARTED')
                for img in minibatch_spec:
                    (img_X, img_Y) = img
                    minibatch_opt, minibatch_cost, minibatch_Y_pred, minibatch_accuracy = self.sess.run([self.opt, self.cost, self.Y_pred, self.accuracy],
                                                                                                         feed_dict = {self.X: img_X, self.Y: img_Y})
                    epoch_cost += minibatch_cost
                    epoch_accuracy += minibatch_accuracy
                print('SPECTOGRAMS APPENDING FOR MINIBATCH --------------------> ENDED')
                epoch_cost = epoch_cost / minibatch_full
                epoch_accuracy = epoch_accuracy / minibatch_full

                if epoch%validation_step == 0:
                    for spec in val_data:
                        (val_X, val_Y) = spec
                        val_loss, val_prediction, val_accuracy = self.sess.run([self.cost, self.Y_pred, self.accuracy],
                                                                                feed_dict = {self.X: val_X, self.Y: val_Y})
                    print(val_accuracy)

                if epoch%checkpoint_step == 0:         
                    if os.path.exists(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'))== False:
                        os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints'),exist_ok=True)
                    self.saver.save(self.sess, os.path.join(os.getcwd(),self.base_dir, self.model_name, 'chekpoints','my-model'))
                       
                if epoch%display_step == 0:
                    print('Cost after ' + str(epoch+1) + ' epoch is: '+ str(epoch_cost))
                    print('Train accuracy is: ' + str(epoch_accuracy))

                if epoch%summary_step == 0:
                    if os.path.exists(os.path.join(os.getcwd(),self.base_dir,self.model_name, 'summaries'))== False:
                        os.makedirs(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'),exist_ok=True)
                        tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)
                    else:
                        tf.summary.FileWriter(os.path.join(os.getcwd(),self.base_dir, self.model_name, 'summaries'), self.sess.graph)

    def test_model(self):
        minibatch_full_test = round(len(self.test_paths) / self.test_batch_size)
        x_test=[]
        y_test=[]
        for i in range(minibatch_full_test):
            x_test_, y_test_= self.data_loader.test_data_loader(i)
            x_test = x_test + x_test_
            y_test = y_test + y_test_
        x_test = tf.stack(x_test, axis = 0)
        y_test = tf.stack(y_test, axis = 0)
        test_cost, test_accuracy  = self.sess.run([self.cost, self.accuracy], feed_dict = {self.X: x_test, self.Y: y_test})
        print("Test accuracy is: " + str(test_accuracy))

                        
    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')
      