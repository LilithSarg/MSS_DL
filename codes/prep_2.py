import numpy as np 
import os
from scipy.io import wavfile
import glob
import random
from utils import *
import pandas as pd

os.chdir('path_to_MUSDB_folder')

splits = os.listdir()
initial_split_dict = dict(flatten(dict(zip(splits, [os.listdir('{}'.format(split)) for split in splits]))))
initial_split_dict_ = dict(zip(initial_split_dict.values(), pd.Series(list(initial_split_dict.keys())).str.split('_', 1, expand = True)[0]))
print('[*] initial split dict: ', initial_split_dict_)
songs = [song for sublist in [os.listdir('{}'.format(split)) for split in splits] for song in sublist]
songs = random.sample(songs, len(songs))

train_perc = .8
train_num = int(train_perc * len(songs))
train_songs = songs[:train_num]
val_songs = list(set(songs) - set(songs).intersection(train_songs))
split_dict = dict(flatten(dict(zip(['train', 'val'], [train_songs, val_songs]))))
split_dict_ = dict(zip(split_dict.values(), pd.Series(list(split_dict.keys())).str.split('_', 1, expand = True)[0]))
print('[*] new split dict: ', split_dict_)
write_dict('../data_prep/split_dict.txt', split_dict)
print('[*] split_dict wrote')

assert set(train_songs).intersection(set(val_songs)) == set(), 'song is both in train and val splits'


print('[*] preprocessing started')

sr = 16000
s_window = 5
ms_window = 50
ms_window_sr  = ms_window * sr // 1000
step = ms_window_sr // 2
stride = 0.6
ms_overlap = vorbis_window(ms_window_sr)

for song in songs:
	print(song)
	m_sr, mix_wav = wavfile.read('{}/{}/mixture16.wav'.format([v for k, v in initial_split_dict_.items() if k == song][0], song))
	v_sr, vocal_wav = wavfile.read('{}/{}/vocals16.wav'.format([v for k, v in initial_split_dict_.items() if k == song][0], song))

	assert m_sr == v_sr == 16000, 'sampling rate is not 16k'
	assert int(len(mix_wav)/sr) == int(len(vocal_wav)/sr), 'stem seconds dont match'
	seconds = int(len(mix_wav)/sr)

		# taking 5 seconds
	for second in range(0, seconds, 5):
		cur_mix_s = mix_wav[int(second * stride * sr): int((second * stride + s_window) * sr)]
		cur_vocal_s = vocal_wav[int(second * stride * sr): int((second * stride + s_window) * sr)]

		ms_count = len(cur_mix_s) // step - 1
		mix_magn = []
		vocal_magn = []
		for ms_ in range(ms_count):
			cur_mix_ms = cur_mix_s[int(ms_ * step) : int(ms_ * step + ms_window_sr)]
			cur_vocal_ms = cur_vocal_s[int(ms_ * step) : int(ms_ * step + ms_window_sr)]

			mix_ms_magn, mix_ms_phase = get_magn_phase(cur_mix_ms, ms_overlap)
			vocal_ms_magn, vocal_ms_phase = get_magn_phase(cur_vocal_ms, ms_overlap)
			mix_magn.append(mix_ms_magn)
			vocal_magn.append(vocal_ms_magn)
		magn_mix_vocal = np.array([mix_magn, vocal_magn])
		np.save('../data_prep/spec/{}/{}_{}_{}.npy'.format([v for k, v in split_dict_.items() if k == song][0], song, int(second*stride), int(second * stride + s_window)), magn_mix_vocal)