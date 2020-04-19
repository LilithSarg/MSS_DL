import numpy as np
import os
# import tensorflow as tf
import glob
from random import randrange
from scipy.io import wavfile

os.chdir('C://Users//lsargsia//OneDrive - Philip Morris International//All files//previous_desktop//ASDS//Deep learning//Project//Music Source Sepration//test_music')

mix_sr, mix_wav =  wavfile.read('Gloria Gaynor - I Will Survive.wav')

ms = 50
real_sr = 16000
window_len  = ms * real_sr // 1000
step = window_len // 2

def vorbis_window(N):
    return np.sin((np.pi / 2) * (np.sin(np.pi * np.array(range(N)) / N)) ** 2)
window_arr = vorbis_window(window_len)

def get_magn_phase(arr):
    fft = np.fft.rfft(arr * window_arr)
    magn = np.abs(fft) + 1e-7
    phase = fft / magn
    return magn, phase

assert mix_sr == real_sr
new_vocal = np.zeros(len(mix_wav))
mask = np.zeros(len(mix_wav))
frame_count = len(mix_wav) // step - 1
for i in range(frame_count):
    mix_cur_window = mix_wav[i * step: i * step + window_len]
    mix_magn, mix_phase = get_magn_phase(mix_cur_window)
    ratio_mask = 200
    mask_clip = ratio_mask * mix_magn
    mask_clip_fft = mask_clip * mix_phase
    mask_window = np.fft.irfft(mask_clip_fft) * vorbis_window(window_len)
    mask[i * step: i * step + window_len] += mask_window

wavfile.write('ratio_mask_test.wav', real_sr, mask.astype("int16"))