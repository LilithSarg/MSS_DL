import numpy as np
from scipy.io import wavfile
import os

# data_path = 'C://Users//lsargsia//OneDrive - Philip Morris International//All files//previous_desktop//ASDS//Deep learning//Project//Music Source Sepration//MUSDBHQ-18_sample//test//Nerve 9 - Pray For The Rain//'
data_path = '../MUSDBHQ-18_sample/test/Nerve 9 - Pray For The Rain'

mix_path = data_path + 'mixture16.wav'
vocal_path = data_path + 'vocals16.wav'
ms = 30
real_sr = 16000
window_len  = ms * real_sr // 1000
step = window_len // 2
mix_sr, mix_wav = wavfile.read(mix_path)
vocal_sr, vocal_wav = wavfile.read(vocal_path)
assert mix_sr == real_sr
assert vocal_sr == real_sr
assert len(mix_wav) == len(vocal_wav)

mix_wav = (mix_wav[:,0] + mix_wav[:,1]) / 2
vocal_wav = (vocal_wav[:,0] + vocal_wav[:,1]) / 2

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
for i in range(frame_count):
    mix_cur_window = mix_wav[i * step: i * step + window_len]
    mix_magn, mix_phase = get_magn_phase(mix_cur_window)
    vocal_cur_window = vocal_wav[i * step: i * step + window_len]
    vocal_magn, vocal_phase = get_magn_phase(vocal_cur_window)

    vocal_new_fft = vocal_magn * mix_phase
    vocal_new_window = np.fft.irfft(vocal_new_fft) * vorbis_window(window_len)
    new_vocal[i * step: i * step + window_len] += vocal_new_window

    mask_clip = np.clip(vocal_magn / mix_magn, 0, 1) * mix_magn
    mask_clip_fft = mask_clip * mix_phase
    mask_window = np.fft.irfft(mask_clip_fft) * vorbis_window(window_len)
    mask[i * step: i * step + window_len] += mask_window

wavfile.write("phase_test.wav", real_sr, new_vocal.astype("int16"))
wavfile.write("ratio_mask_test.wav", real_sr, mask.astype("int16"))