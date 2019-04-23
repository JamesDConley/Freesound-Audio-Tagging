from scipy.io import wavfile
from scipy.signal import resample
import librosa, glob
import pandas as pd
import numpy as np
import ezPickle as p
from scipy.misc import imsave


print("Please enter name of input folder")
in_folder = input()
print("Please enter name of output folder")
out_folder = input()
le = p.load('le')
data = []
max_len = 0
new_rate = 16000
for file_name in glob.glob(in_folder+"/*"):
	print(file_name)
	rate, data = wavfile.read(file_name)
	print("\t read")
	print('\t',data.shape, " Frames at ", rate)
	new_data = resample(data,  round(data.shape[0]/rate * new_rate))
	spectrogram = librosa.feature.melspectrogram(y=new_data.astype(float), sr=new_rate)
	print("\t sampled")
	print('\t',spectrogram.shape)
	imsave(out_folder+'/' + file_name[file_name.index('/'):] +".png", spectrogram)
	print("\twrote")
	if spectrogram.shape[1] > max_len:
		max_len = spectrogram.shape[1]
		max_shape = spectrogram.shape
print(max_len)
print(max_shape)
