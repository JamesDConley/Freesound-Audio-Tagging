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
p.save(le,'le')
data = []
max_len = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

for file_name in files:
	#print(file_name)
	
	spectrogram = imread("mel_spec_curated/" + file_name +".png")
	#print("\t sampled")
	#print('\t',spectrogram.shape)
	if spectrogram.shape[1] > max_len:
		max_len = spectrogram.shape[1]
		max_shape = spectrogram.shape
print(max_len)
print(max_shape)

def audio_generator(sounds_per_batch):
	while True:
		data_list = []
		label_list = []
		for i in range(len(files)):
			file_name = files[i]
			data_list.append(np.true_divide(np.transpose((imread("mel_spec_curated/" + file_name +".png"))),256))
			temp = [0]*len(le.classes_.tolist())
			for item in labels[i].split(','):
				temp[le.transform([item])[0]] = 1
			label_list.append(temp.copy())
			if len(data_list) == sounds_per_batch:
				#print(data_list)
				#print(label_list)
				x = np.array(pad_sequences(data_list, maxlen = max_len, value=-1)).reshape(sounds_per_batch, max_len, 128)
				y = np.array(label_list)
				print(x.shape)
				print(y.shape)
				yield x , y 
				data_list = []
				label_list = []

batch_size = 51

model = Sequential()
model.add(Masking(mask_value=0, input_shape=(max_len, 128)))
model.add(LSTM(256,  return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128,  return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(len(le.classes_.tolist()),  activation='sigmoid'))

rms = RMSprop()

p_model = multi_gpu_model(model, gpus=3)

p_model.compile(loss='mean_squared_error',optimizer=rms, metrics=['mean_squared_error'])


p_model.fit_generator(audio_generator(batch_size), epochs=100,  verbose=1,  shuffle=False, steps_per_epoch=math.ceil(len(files)/batch_size),max_queue_size=1)
p.save(p_model, 'p_model_100')
