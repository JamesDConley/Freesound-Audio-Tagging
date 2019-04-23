from scipy.io import wavfile
from scipy.signal import resample
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras import backend as K
from keras.layers import Dense, LSTM, Dropout, Masking
from keras.optimizers import RMSprop
from keras.activations import softmax
from keras.utils import multi_gpu_model
import keras, math, os, glob
import pandas as pd
import numpy as np
import ezPickle as p


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

max_len = 0
max_shape = 0
sequences = []

train_curated = pd.read_csv('train_curated.csv')
files = train_curated['fname']
labels = train_curated['labels']

new_rate = 100
"""
data = []
for file_name in files:
	print(file_name)
	rate, data = wavfile.read('train_curated/tiny_resampled_'+file_name)
	print("\t read")
	print('\t',data.shape, " Frames at ", rate)
	new_data = resample(data,  round(data.shape[0]/rate * new_rate))
	print("\t sampled")
	print('\t',new_data.shape, " Frames at ", new_rate)
	wavfile.write('train_curated/tiny_resampled_'+file_name, new_rate, new_data)
	print("\twrote")
	if new_data.shape[0] > max_len:
		max_len = new_data.shape[0]
		max_shape = new_data.shape
#import matplotlib.pyplot as plt
#plt.plot(data)
#plt.ylabel('audio')
#plt.show()"""
max_len = 0
for i in range(len(files)):
	file_name = 'train_curated/tiny_resampled_' + files[i]
	rate, data = wavfile.read(file_name)
	if data.shape[0] > max_len:
		max_len = data.shape[0]
print(max_len)

from sklearn import preprocessing
le =  preprocessing.LabelEncoder()
print("here")
le.fit(labels)
print("there")
def audio_generator(sounds_per_batch):
	while True:
		data_list = []
		label_list = []
		for i in range(len(files)):
			file_name = 'train_curated/tiny_resampled_' + files[i]
			rate, data = wavfile.read(file_name)
			data_list.append(np.array(data).reshape(len(data),1))
			temp = [0]*len(le.classes_.tolist())
			temp[le.transform([labels[i]])[0]] = 1
			label_list.append(temp.copy())
			if len(data_list) == sounds_per_batch:
				#print(data_list)
				#print(label_list)
				x = np.array(pad_sequences(data_list, maxlen = max_len)).reshape(sounds_per_batch, max_len, 1)
				y = np.array(label_list)
				print(x.shape)
				print(y.shape)
				yield x , y 
				data_list = []
				label_list = []
			
		
#for x, y in audio_generator(10):
#	print(x, y)
#	os.sleep(5)
batch_size = 9

model = Sequential()
model.add(Masking(mask_value=0, input_shape=(max_len, 1)))
model.add(LSTM(256,  return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128,  return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(len(le.classes_.tolist()),  activation='softmax'))

rms = RMSprop()

p_model = multi_gpu_model(model, gpus=3)

p_model.compile(loss='categorical_crossentropy',optimizer=rms, metrics=['categorical_accuracy'])


p_model.fit_generator(audio_generator(batch_size), epochs=10,  verbose=1,  shuffle=False, steps_per_epoch=math.ceil(len(files)/batch_size),max_queue_size=1)
p.save(p_model, 'p_model')
#sequence = pad_sequences(sequences, maxlen=max_len, dtype='int32', padding='pre', truncating='pre', value=-1)
