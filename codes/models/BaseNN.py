import tensorflow as tf
from data_loader import *
from abc import abstractmethod
import random
import math
import itertools
from utils import *

class BaseNN:
    def __init__(self, train_spec_dir, val_spec_dir, test_spec_dir, num_epochs, train_batch_size,
                 val_batch_size, test_batch_size, sequence_length, fft_length, 
                 learning_rate, base_dir, max_to_keep, model_name):

        self.data_loader = DataLoader(train_spec_dir, val_spec_dir, test_spec_dir, train_batch_size, 
                val_batch_size, test_batch_size, sequence_length, fft_length)

        self.train_paths = glob.glob(os.path.join(train_spec_dir, '*.npy'), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_spec_dir, '*.npy'), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_spec_dir, '*.npy'), recursive=True)
        
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
        self.summary_dir = os.path.join(base_dir, 'summary')
        self.checkpoint_dir =  os.path.join(base_dir, 'checkpoint')

    def create_network(self):            # 100             199              401
        self.X = tf.placeholder('float', [None, self.sequence_length, self.fft_length], name = 'X')
        self.Y = tf.placeholder('float', [None, self.sequence_length, self.fft_length], name = 'Y')
        self.Y_pred = self.network_bgru(batch_x = self.X, seq_len = self.train_batch_size * [self.sequence_length])
        # self.Y_pred = self.network_d5(self.X)
        self.cost = self.metrics(self.Y, self.Y_pred)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')  # number of batches
        self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost, global_step = self.global_step)
        
    def initialize_network(self):
            print("[*] Initializing network...")
            self.sess = tf.Session()
            self.summary_op = tf.summary.merge_all()   # merges all summaries : val + train
            self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
            if self.summary_dir != "":
                self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.summary_dir, "train"), self.sess.graph)
                self.val_summary_writer = tf.summary.FileWriter(os.path.join(self.summary_dir,"validation"), self.sess.graph)

            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if checkpoint:
                print("[*] Restoring from checkpoint...")
                checkpoint_path = checkpoint.model_checkpoint_path
                self.saver.restore(self.sess, checkpoint_path)
            else:
                self.sess.run(tf.global_variables_initializer())

    def train_model(self, display_step, validation_step, checkpoint_step, summary_step):
            minibatch_full = round(len(self.train_paths) / self.train_batch_size)
            minibatch_full_v = round(len(self.val_paths) / self.val_batch_size)
            print('Train minibatches --> {}'.format(minibatch_full))
            print('Validation minibatches --> {}'.format(minibatch_full_v))

            for epoch in range(self.num_epochs):
                print('[*] Epoch --> {}'.format(epoch))
                random.shuffle(self.data_loader.train_paths)
                random.shuffle(self.data_loader.val_paths)
                for k_th_batch in range(minibatch_full):
                    train_matrix, train_label = self.data_loader.train_data_loader(k_th_batch)
                    train_pred, minibatch_opt, train_minibatch_cost, train_summary, global_step = self.sess.run([self.Y_pred, self.opt, self.cost, self.summary_op, self.global_step], 
                                                                                  feed_dict = {self.X: train_matrix, self.Y: train_label})
                    print('Global step --> {}'.format(global_step))
                    if global_step % validation_step == 0:
                        k_th_batch_val = int(k_th_batch % minibatch_full_v)
                        val_matrix, val_label = self.data_loader.val_data_loader(k_th_batch_val)
                        val_minibatch_cost, val_summary = self.sess.run([self.cost, self.summary_op], 
                                                                         feed_dict = {self.X: val_matrix, self.Y: val_label})

                        self.val_summary_writer.add_summary(val_summary, global_step)
                        print('Validation loss -- > {}'.format(val_minibatch_cost))

                    if global_step % summary_step == 0:
                        self.train_summary_writer.add_summary(train_summary, global_step)
                        print('Summary done')

                    if global_step % checkpoint_step == 0:
                        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name + ".ckpt"), global_step=global_step)
                        print('Checkpoint save done')

                    if global_step % display_step == 0:
                        print('Train loss --> {}'.format(train_minibatch_cost))

    def test_model(self):
        ms = 50
        real_sr = 16000
        window_sec = 5
        window_input =  real_sr * window_sec
        window = real_sr * ms // 1000
        step = window // 2
        vorbis_window_ = vorbis_window(window)
        song_names, song_wav = self.data_loader.test_data_loader()
        for song_i, mix_wav in enumerate(song_wav):
            output_wav = np.zeros(len(mix_wav))
            window_input_count = len(mix_wav) // window_input
            window_ms = window_input // step - 1
            for i in range(window_input_count):
                new_wav_5 = np.zeros(window_input)
                mix_cur_data = mix_wav[i * window_input: (i + 1) * window_input]
                inp = []
                fft = []
                for j in range(window_ms):
                    mix_cur_data_ms = mix_cur_data[step * j : step * j + window]
                    cur_fft = np.fft.rfft(mix_cur_data_ms)
                    magn = np.abs(cur_fft)
                    inp.append(np.array(magn))
                    fft.append(cur_fft)
                inp = np.array(inp)[np.newaxis, ...]
                ratio_mask = np.clip(self.sess.run(self.Y_pred, feed_dict = {self.X: inp}), 0, 1)
                new_fft = fft * np.round(ratio_mask) # * (fft/inp)
                new_mix_cur_data = np.fft.irfft(new_fft) * vorbis_window_
                for j in range(new_mix_cur_data.shape[1]):
                    new_wav_5[step * j : step * j + window] += new_mix_cur_data[0][j]
                output_wav[i * window_input: (i + 1) * window_input] += new_wav_5
            wavfile.write(os.path.join(self.base_dir, 'Separated_{}'.format(song_names[song_i])), real_sr, output_wav.astype("int16"))
            print(song_names[song_i], 'DONE')
        gt, pred = self.data_loader.test_data_loader_sdr()
        assert len(gt) == len(pred)
        window_input_count = len(gt) // window_input
        window_ms = window_input // step - 1
        sdr_ = []
        for i in range(window_input_count):
            gt_cur_data = gt[i * window_input: (i + 1) * window_input]
            pred_cur_data = pred[i * window_input: (i + 1) * window_input]
            for j in range(window_ms):
                sdr_ms = []
                gt_cur_data_ms = gt_cur_data[step * j : step * j + window]
                pred_cur_data_ms = pred_cur_data[step * j : step * j + window]
                gt_cur_fft = np.fft.rfft(gt_cur_data_ms)
                pred_cur_fft = np.fft.rfft(pred_cur_data_ms)
                gt_magn = np.abs(gt_cur_fft)
                pred_magn = np.abs(pred_cur_fft)
                diff = np.absolute(gt_magn - pred_magn)
                SDR = 10*np.log10( np.mean(np.absolute(gt_magn**2)) / np.mean(diff**2) + 10e-7)
                sdr_ms.append(SDR)
            sdr_.append(sdr_ms)
        from scipy import stats
        sdr_ = np.array(sdr_)
        print(sdr_)
        print('Shape -->', sdr_.shape)
        sdr_f = np.clip(sdr_[~np.isnan(sdr_)], 0, 100)
        print(sdr_f)
        print('Shape nan removed -->', sdr_f.shape)
        print('Mean SDR -->', np.mean(sdr_f))
        print('Median SDR -->', np.median(sdr_f))
        print('Mode SDR -->', stats.mode(sdr_f))
                        
    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')
      