import os
import glob
import numpy as np
from random import randrange
from scipy.io import wavfile
class DataLoader:
    def __init__(self, train_spec_dir, val_spec_dir, test_spec_dir, train_batch_size, val_batch_size, 
            test_batch_size, sequence_length, fft_length):

        base_dir = ''
        self.train_paths = glob.glob(os.path.join(train_spec_dir, '*.npy'), recursive = True)
        self.val_paths = glob.glob(os.path.join(val_spec_dir, '*.npy'), recursive = True)
        self.test_paths = glob.glob(os.path.join(test_spec_dir, '**/*.wav'), recursive = True)
        self.test_paths_voc_gt = glob.glob(os.path.join(test_spec_dir, 'labeled/*vocals16.wav'), recursive = True)
        self.test_paths_pred = glob.glob(os.path.join(base_dir, '*mixture16.wav'), recursive = True)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.sequence_length = sequence_length
        self.fft_length = fft_length
        self.mean_, self.std_ = np.load('data_prep/mean_std.npy')

    # takes .npy file where first sublist is mixture magnitude, second subset is vocal magnitude      
    def load_sec_mpa(self, path): 
        matrix, label = np.load(path)
        matrix_n = (matrix - self.mean_) / self.std_
        rm1 = label / (matrix_n + 10**(-6))
        rm2 = np.sqrt(label ** 2 / (matrix_n ** 2 + label ** 2))
        ratio_mask = np.clip(rm2, 0, 1)
        return matrix_n, ratio_mask

    def batch_data_loader(self, batch_size, file_paths, index):
        matrixs = []
        labels = []   
        for spec in file_paths[index*batch_size : (index+1)*batch_size]:
            matrix, label = self.load_sec_mpa(spec)
            matrixs.append(matrix)
            labels.append(label)
        return matrixs, labels

    def train_data_loader(self, index):
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index)

    def val_data_loader(self, index):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index)

    def test_data_loader(self):
        song_names = [self.test_paths[i].split('/')[-1] for i in range(len(self.test_paths))]
        song_wav = [wavfile.read(self.test_paths[i])[1] for i in range(len(self.test_paths))]
        return song_names, song_wav

    def test_data_loader_sdr(self):
        s, vocal_wav_gt = wavfile.read(self.test_paths_voc_gt[0])
        s, vocal_wav_pred = wavfile.read(self.test_paths_pred[0])
        return vocal_wav_gt, vocal_wav_pred