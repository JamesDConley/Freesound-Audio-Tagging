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
le = p.load('le')
train_curated = pd.read_csv('train_curated.csv')
files = train_curated['fname']
labels = train_curated['labels']

p_model = p.load('p_model')
max_len = 1800
correct = 0
for i in range(len(files)):
	#print(file_name)
	file_name = files[i]
	spectrogram = imread("mel_spec_curated/" + file_name +".png")
	x = np.array(pad_sequences([np.true_divide(np.transpose((spectrogram)),256)], maxlen = max_len, value=-1)).reshape(1, max_len, 128)
	pred = p_model.predict(x).tolist()[0]
	p_model.reset_states()
	p_num = pred.index(max(pred))
	category = le.inverse_transform([p_num])[0]
	if category in labels[i].split(','):
		correct+=1
	print(category, labels[i].split(','))
	
	#print("\t sampled")
	#print('\t',spectrogram.shape)
print(correct/len(files))
	
