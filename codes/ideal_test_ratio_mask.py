import numpy as np
from scipy.io import wavfile
import os

data_path = ''
os.chdir(data_path)
mix_path = os.path.join('mixture16.wav')
vocal_path = os.path.join('vocals16.wav')
ms = 50
real_sr = 16000
window_len  = ms * real_sr // 1000
step = window_len // 2
mix_sr, mix_wav = wavfile.read('mixture16.wav')
vocal_sr, vocal_wav = wavfile.read('vocals16.wav')
assert mix_sr == real_sr
assert vocal_sr == real_sr
assert len(mix_wav) == len(vocal_wav)

def vorbis_window(N):
    return np.sin((np.pi / 2) * (np.sin(np.pi * np.array(range(N)) / N)) ** 2)

window_arr = vorbis_window(window_len)

def get_magn_phase(arr):
    fft = np.fft.rfft(arr * window_arr)
    magn = np.abs(fft) + 1e-7
    phase = fft / magn
    return magn, phase

new_vocal = np.zeros(len(vocal_wav))
mask = np.zeros(len(vocal_wav))
frame_count = len(vocal_wav) // step - 1
print('frame_count: ', frame_count)
print('window_len: ', window_len)

for i in range(frame_count):
    mix_cur_window = mix_wav[i * step: i * step + window_len]
    mix_magn, mix_phase = get_magn_phase(mix_cur_window)
    vocal_cur_window = vocal_wav[i * step: i * step + window_len]
    vocal_magn, vocal_phase = get_magn_phase(vocal_cur_window)

    vocal_new_fft = vocal_magn * mix_phase
    vocal_new_window = np.fft.irfft(vocal_new_fft) * vorbis_window(window_len)
    new_vocal[i * step: i * step + window_len] += vocal_new_window

    ratio_mask = np.clip(vocal_magn / mix_magn, 0, 1)
    mask_clip = ratio_mask * mix_magn
    mask_clip_fft = mask_clip * mix_phase
    mask_window = np.fft.irfft(mask_clip_fft) * vorbis_window(window_len)
    mask[i * step: i * step + window_len] += mask_window

wavfile.write(os.path.join('phase_test.wav'), real_sr, new_vocal.astype("int16"))
wavfile.write(os.path.join('ratio_mask_test.wav'), real_sr, mask.astype("int16"))