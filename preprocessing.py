import numpy as np
from scipy.io import wavfile
import os

ms = 50
real_sr = 16000
window_sec = 5
window_len  = ms * real_sr // 1000
step = window_len // 2
stems = ['mixture', 'vocals']
folders = ['train', 'val']
stride = 0.95


def read_signal(path):
    sr, wav = wavfile.read(path)
    sec = int(np.round(len(wav)/sr, decimals = 0))  ## fixme: no averaging, use ffmpeg... ac 1 DONE
    return sr, wav, sec

def vorbis_window(N):
    return np.sin((np.pi / 2) * (np.sin(np.pi * np.array(range(N)) / N)) ** 2)

def get_magn_phase(arr):
    fft = np.fft.rfft(arr * window_arr)
    magn = np.abs(fft) + 1e-7
    phase = fft / magn
    return magn, phase

def mk_stem_fft_dir(list_):
    for stem in list_:
        os.system('mkdir ' + stem)
    pass

def frame_fft_sa(cur_data_, frame_count_, window_len_, step_):
    sec_fft = []
    for i in range(frame_count_):
        cur_window = cur_data_[i * step_: i * step_ + window_len_]
        magn, phase = get_magn_phase(cur_window)
        window_data = np.hstack((magn, phase))
        sec_fft.append(window_data)
    return sec_fft

os.chdir('../MUSDB18-HQ')
for split in folders:
    os.chdir('{}'.format(split))
    for song in os.listdir(os.getcwd()):
        os.chdir('{}'.format(song))
        mk_stem_fft_dir(['{}_{}'.format(stem, 'prep') for stem in stems])
        mix_sr, mix_wav, mix_sec = read_signal('{}{}'.format(stems[0], '16.wav'))
        vocal_sr, vocal_wav, vocal_sec = read_signal('{}{}'.format(stems[1], '16.wav'))
        assert mix_sr == real_sr
        assert vocal_sr == real_sr
        # assert len(mix_wav) == len(vocal_wav)
        assert len(mix_wav) / mix_sr == len(vocal_wav) / vocal_sr
        assert mix_sec == vocal_sec
        seconds_count = mix_sec
        sr = mix_sr
        window_arr = vorbis_window(window_len)
        print(song)
        print('DATA_POINTS --> ', len(mix_wav))
        print('SECONDS --> ', len(mix_wav)/16000)
        for second in range(0, seconds_count*(1+int(stride))):
            mix_sec_fft = []
            vocal_sec_fft = []
            mix_cur_data = mix_wav[int(second * stride * sr): int((second * stride) * sr + window_sec * sr)]
            vocal_cur_data = vocal_wav[int(second * stride * sr): int((second * stride) * sr + window_sec * sr)]
            mix_frame_count = len(mix_cur_data) // step - 1
            vocal_frame_count = len(vocal_cur_data) // step - 1
            assert mix_frame_count == vocal_frame_count
            frame_count = mix_frame_count
            mix_sec_fft, vocal_sec_fft = \
                            [ frame_fft_sa(cur_data_, frame_count, window_len, step) for cur_data_ in [mix_cur_data, vocal_cur_data] ]

            np.save('./{}/{}_mixture_MP.npy'.format('mixture_prep', str(second)), mix_sec_fft)
            np.save('./{}/{}_vocals_MP.npy'.format('vocals_prep', str(second)), vocal_sec_fft)
        os.chdir('../')
    os.chdir('../')