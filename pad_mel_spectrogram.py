from scipy.io import wavfile

import librosa, glob
import pandas as pd
import numpy as np
import ezPickle as p
from scipy.misc import imsave, imread



max_len = 0
max_len = 1800
for file_name in glob.glob(input("Enter folder name\n")+'/*'):
	print(file_name)
	spectrogram = imread(file_name)
	print("\t read")
	spectrogram = np.pad(spectrogram, ((0,0),(0,max_len - spectrogram.shape[1])), 'constant')
	
	print("\t padded")
	print('\t',spectrogram.shape)
	imsave(file_name, spectrogram)
	print("\twrote")
	if spectrogram.shape[1] > max_len:
		max_len = spectrogram.shape[1]
		max_shape = spectrogram.shape
print(max_len)

