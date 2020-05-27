import numpy as np
from scipy.io import wavfile
import os
import json
import _pickle as pickle

def write_dict(path, file):
	with open(path, 'w') as data:
	    data.write(str(file))

def vorbis_window(N):
    return np.sin((np.pi / 2) * (np.sin(np.pi * np.array(range(N)) / N)) ** 2)

def get_magn_phase(arr, overlap_window):
    fft = np.fft.rfft(arr * overlap_window)
    magn = np.abs(fft) + 1e-7
    phase = fft / magn
    return magn, phase

def flatten(d,sep="_"):
    import collections

    obj = collections.OrderedDict()

    def recurse(t,parent_key=""):
        
        if isinstance(t,list):
            for i in range(len(t)):
                recurse(t[i],parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t,dict):
            for k,v in t.items():
                recurse(v,parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)

    return obj