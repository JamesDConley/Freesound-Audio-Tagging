from scipy.io import wavfile
from scipy.signal import resample
import librosa, glob
import pandas as pd
import numpy as np
import ezPickle as p
from scipy.misc import imsave

df = pd.DataFrame()
print("Please enter name of input folder")
in_folder = input()
print("Please enter name of output folder")
out_folder = input()
le = p.load('le')
data = []
lengths = []
count = 0
files = glob.glob(in_folder+"/*")
for file_name in files :
	print(file_name)
	rate, data = wavfile.read(file_name)
	print("\t read")
	print('\t',data.shape, " Frames at ", rate)
	print('\t', count/len(files))
	#new_data = resample(data,  round(data.shape[0]/rate * new_rate))
	spectrogram = librosa.feature.melspectrogram(y=data.astype(float), sr=rate)
	print("\t Converted")
	print('\t',spectrogram.shape)
	imsave(out_folder+'/' + file_name[file_name.index('/'):] +".png", spectrogram)
	print("\twrote")
	lengths.append(spectrogram.shape[1])
	count+=1
series = pd.Series(lengths)
p.save(series, 'series')
print(series.describe())
