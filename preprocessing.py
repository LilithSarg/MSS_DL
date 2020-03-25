import numpy as np
from scipy.io import wavfile
import os

ms = 50
sr = 16000
window_sec = 5
stride = 0.6
window_len  = ms * sr // 1000
step = window_len // 2
stems = ['mixture', 'vocals']
folders = ['train', 'val']
dur_mismatch = ['Matthew Entwistle - Dont You Ever']
bad_songs = ['Music Delta - Country2', 'Music Delta - 80s Rock', 'Music Delta - Gospel', 'Music Delta - Britpop']


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
    return magn

def mk_stem_fft_dir(list_):
    for stem in list_:
        os.system('mkdir ' + stem)
    pass

def frame_fft_sa(cur_data_, frame_count_, window_len_, step_):
    sec_fft = []
    for i in range(frame_count_):
        cur_window = cur_data_[i * step_: i * step_ + window_len_]
        magn = get_magn_phase(cur_window)
        window_data = np.hstack(magn)  # keep only
        sec_fft.append(window_data)
    return sec_fft

os.chdir('MUSDB18-HQ')
mk_stem_fft_dir(['spectograms'])
for split in folders:
    os.chdir('{}'.format(split))
    for song in list(set(os.listdir(os.getcwd())) - set(dur_mismatch)):
    # for song in bad_songs:
        os.chdir('{}'.format(song))
        mix_sr, mix_wav, mix_sec = read_signal('{}{}'.format(stems[0], '16.wav'))
        vocal_sr, vocal_wav, vocal_sec = read_signal('{}{}'.format(stems[1], '16.wav'))
        assert mix_sr == vocal_sr == sr
        # assert len(mix_wav) == len(vocal_wav)
        seconds_count = int(np.round(len(mix_wav)/sr, decimals = 0))
        window_arr = vorbis_window(window_len)
        print(song)
        print('DATA_POINTS --> ', len(mix_wav))
        print('SECONDS --> ', len(mix_wav)/16000)
        for second in range(0, int((1 + stride) * seconds_count), 5):
            mix_sec_fft = []
            vocal_sec_fft = []
            mix_cur_data = mix_wav[int(second * stride * sr): int((second * stride + window_sec) * sr)] # 0-5, 3 - 8
            vocal_cur_data = vocal_wav[int(second * stride * sr): int((second * stride + window_sec) * sr)]
            
            # print(second, second * 0.6 * sr, (second * 0.6 + window_sec) * sr )

            mix_frame_count = len(mix_cur_data) // step - 1
            vocal_frame_count = len(vocal_cur_data) // step - 1
            frame_count = mix_frame_count
            mix_sec_fft, vocal_sec_fft = \
                            [ frame_fft_sa(cur_data_, frame_count, window_len, step) for cur_data_ in [mix_cur_data, vocal_cur_data] ]
            if np.array(mix_sec_fft).shape == np.array(vocal_sec_fft).shape == (199, 401):
              all_stem_magn = np.reshape(np.concatenate((mix_sec_fft, vocal_sec_fft)), (2, 199, 401))
              cur_file = '{}_{}_{}_SPEC.npy'.format(int(second * stride), int(second * stride + window_sec), song)
              np.save('../../{}/{}/{}_{}_{}_SPEC.npy'.format('spectograms', split, int(second * stride), int(second * stride + window_sec), song), all_stem_magn)
        os.chdir('../')
    os.chdir('../')