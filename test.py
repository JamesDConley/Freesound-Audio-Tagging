from scipy.io import wavfile
from scipy.signal import resample
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras import backend as K
from keras.layers import Dense, LSTM, Dropout, Masking
from keras.optimizers import RMSprop
from keras.activations import softmax
from keras.utils import multi_gpu_model
from sklearn import preprocessing
import keras, math, os, glob
import pandas as pd
import numpy as np
import ezPickle as p

from scipy.misc import imread


train_curated = pd.read_csv('train_curated.csv')
files = train_curated['fname']
labels = train_curated['labels']

labels = [item for label_list in labels for item in label_list.split(',')]
le =  preprocessing.LabelEncoder()
le.fit(labels)
print(len(le.classes_.tolist()))
data = []
max_len = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

for file_name in files:
	#print(file_name)
	
	spectrogram = imread("mel_spec_curated/" + file_name +".png")
	#print("\t sampled")
	#print('\t',spectrogram.shape)
	print(np.true_divide(spectrogram,256))
	
	if spectrogram.shape[1] > max_len:
		max_len = spectrogram.shape[1]
		max_shape = spectrogram.shape
print(max_len)
print(max_shape)
