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
        self.Y_pred = self.network(self.X)
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
                    train_matrix, train_label = np.array(train_matrix), np.array(train_label)
                    minibatch_opt, train_minibatch_cost, train_summary, global_step = self.sess.run([self.opt, self.cost, self.summary_op, self.global_step], 
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
        def vorbis_window(N):
            return np.sin((np.pi / 2) * (np.sin(np.pi * np.array(range(N)) / N)) ** 2)
        def get_magn_phase(arr):
            fft = np.fft.rfft(arr * window_arr)
            magn = np.abs(fft) + 1e-7
            phase = fft / magn
            return magn, phase
        def frame_fft_sa(cur_data_, frame_count_, window_len_, step_):
            sec_fft = []
            sec_phase = []
            for i in range(frame_count_):
                cur_window = cur_data_[i * step_: i * step_ + window_len_]
                magn, phase = get_magn_phase(cur_window)
                sec_fft.append(np.hstack(magn))
                sec_phase.append(np.hstack(phase))
            return sec_fft, sec_phase

        mix_wav = self.data_loader.test_data_loader()
        ms = 50
        real_sr = 16000
        window_len  = ms * real_sr // 1000
        step = window_len // 2
        stride = 0.6
        window_sec = 5

        mask = np.zeros(len(mix_wav))
        frame_count = len(mix_wav) // step - 1
        seconds_count = int(np.round(len(mix_wav)/real_sr, decimals = 0))
        window_arr = vorbis_window(window_len)
        for second in range(0, int((1 + stride) * seconds_count), 5):
            mix_cur_data = mix_wav[int(second * stride * real_sr): int((second * stride + window_sec) * real_sr)]
            frame_count = len(mix_cur_data) // step - 1
            self.mix_sec_fft, self.mix_sec_phase = frame_fft_sa(mix_cur_data, frame_count, window_len, step)
            print(type(self.mix_sec_fft))
            print(self.Y_pred.shape)
            exit()
            mask_clip = tf.mulmul(self.Y_pred, self.mix_sec_fft)
            mask_clip_fft = mask_clip * mix_sec_phase
            mask_window = np.fft.irfft(mask_clip_fft) * vorbis_window(window_len)
            mask[int(second * stride * real_sr): int((second * stride + window_sec) * real_sr)] += mask_window

        wavfile.write(os.path.join(base_dir, 'ratio_mask_test.wav'), real_sr, mask.astype("int16"))

                        
    @abstractmethod
    def network(self, X):
        raise NotImplementedError('subclasses must override network()!')

    @abstractmethod
    def metrics(self, Y, y_pred):
        raise NotImplementedError('subclasses must override metrics()!')
      