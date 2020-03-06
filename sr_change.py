import os
import subprocess

folders = ['val_art']
os.chdir('data_sample')
for split in folders:
	os.chdir('{}'.format(split))
	for song in os.listdir(os.getcwd()):
		os.chdir('{}'.format(song))
		print(os.listdir(os.getcwd()))
		subprocess.call(['ffmpeg -i vocals.wav -ar 16000 -ac 1 vocals16.wav'], shell = True)
		subprocess.call(['ffmpeg -i mixture.wav -ar 16000 -ac 1 mixture16.wav', 'mixture16.wav'], shell = True)
		os.chdir('../')
	os.chdir('../')