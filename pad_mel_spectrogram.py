from scipy.io import wavfile

import librosa, glob
import pandas as pd
import numpy as np
import ezPickle as p
from scipy.misc import imsave, imread



max_len = 0
max_len = 500
files = glob.glob(input("Enter folder name\n")+'/*')
count = 0
for file_name in files:
	print(file_name)
	spectrogram = imread(file_name)
	print("\t read")
	spectrogram = spectrogram[:,:max_len]
	print(spectrogram.shape)
	spectrogram = np.pad(spectrogram, ((0,0),(0,max_len - spectrogram.shape[1])), 'constant')
	
	print("\t padded")
	print('\t',spectrogram.shape)
	imsave(file_name, spectrogram)
	print("\twrote")
	if spectrogram.shape[1] > max_len:
		max_len = spectrogram.shape[1]
		max_shape = spectrogram.shape
	print(count/len(files))
	count+=1	
print(max_len)

