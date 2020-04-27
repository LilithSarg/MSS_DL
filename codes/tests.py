import numpy as np
import os
# import tensorflow as tf
import glob
from random import randrange
from scipy.io import wavfile

os.chdir(r'C:\Users\lsargsia\OneDrive - Philip Morris International\All files\previous_desktop\ASDS\Deep learning\Project\Music Source Sepration\data_results\spectograms')

folders = ['train', 'val']

for split in folders:
	spec_list = []
	spec_name = []
	for spec in os.listdir(split):
		spec_list.append(np.load(os.path.join(os.getcwd(), split, spec)))
		spec_name.append(spec)
	for i in range(len(spec_list)):
		if np.array_equal(spec_list[i], spec_list[-1]):
			print(spec_name[i], '=====', spec_name[-1])
	# print(len(os.listdir(split)))
