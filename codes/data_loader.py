import os
import glob
import numpy as np
from random import randrange

class DataLoader:

    def __init__(self, train_spec_dir, val_spec_dir, test_spec_dir, train_batch_size, val_batch_size, 
            test_batch_size, sequence_length, fft_length):

        self.train_paths = glob.glob(os.path.join(train_spec_dir, '*.npy'), recursive=True)
        self.val_paths = glob.glob(os.path.join(val_spec_dir, '*.npy'), recursive=True)
        self.test_paths = glob.glob(os.path.join(test_spec_dir, '*.npy'), recursive=True)

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.sequence_length = sequence_length
        self.fft_length = fft_length

    #load_image
    def load_sec_mpa(self, path):    # takes .npy file where first sublist is mixture magnitude, second subset is vocal magnitude
        matrix = np.load(path)[0]    # loads only magn mix
        label  = np.load(path)[1]    # loads only magn vocal 
        return matrix, label

    def batch_data_loader(self, batch_size, file_paths, index):
        matrix = []
        labels = []   
        for spec in file_paths[index*batch_size : (index+1)*batch_size]:
            matrix.append(self.load_sec_mpa(spec)[0])
            labels.append(self.load_sec_mpa(spec)[1])
        return matrix, labels

    def train_data_loader(self, index):
        return self.batch_data_loader(self.train_batch_size, self.train_paths, index)
      
    def val_data_loader(self, index):
        return self.batch_data_loader(self.val_batch_size, self.val_paths, index)

    def test_data_loader(self, index):
        return self.batch_data_loader(self.test_batch_size, self.test_paths, index)


