from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adadelta
from keras.utils import multi_gpu_model
import math, os, keras
import pandas as pd
import numpy as np
from scipy.misc import imread
import ezPickle as p
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
train_curated = pd.read_csv('train_curated.csv')
train_noisy = pd.read_csv('train_noisy.csv')
files = train_curated['fname']
labels = train_curated['labels']
n_files = train_noisy['fname']
n_labels = train_noisy['labels']
le = p.load('le')
max_len = 0
cb = keras.callbacks.TensorBoard(histogram_freq=0)
for i in range(len(files)):
	file_name = files[i]
	spectrogram = imread("trimmed_curated_spec/" + file_name +".png")
	if max_len < spectrogram.shape[1]:
		max_len = spectrogram.shape[1]
print(max_len)
def generator(batch_size):
	input_batch = []
	output_batch = []
	while True:
		for i in range(len(files)):
			file_name = files[i]
			spectrogram = imread("trimmed_curated_spec/" + file_name +".png")
			
			label = labels[i]

			input_batch.append(spectrogram)
			temp = [0]*len(le.classes_.tolist())
			for item in le.transform(label.split(',')):
				temp[item] = 1
			output_batch.append(temp.copy())
			if len(input_batch) == batch_size:
				yield np.array(input_batch.copy()).reshape(batch_size, 128, max_len, 1), np.array(output_batch.copy())
				input_batch = []
				output_batch = []
def noisy_generator(batch_size):
	input_batch = []
	output_batch = []
	while True:
		for i in range(len(files)):
			file_name = n_files[i]
			spectrogram = imread("trimmed_noisy_spec/" + file_name +".png")
			
			label = n_labels[i]

			input_batch.append(spectrogram)
			temp = [0]*len(le.classes_.tolist())
			for item in le.transform(label.split(',')):
				temp[item] = 1
			output_batch.append(temp.copy())
			if len(input_batch) == batch_size:
				yield np.array(input_batch.copy()).reshape(batch_size, 128, max_len, 1), np.array(output_batch.copy())
				input_batch = []
				output_batch = []
def both_generator(batch_size):
	g = generator(batch_size)
	
	ng = noisy_generator(batch_size)
	parity = True
	while True:
		if parity:
			yield next(g)
		else:
			yield next(ng)
		parity = not parity
		
model = Sequential()
model.add(Conv2D(32, 3, strides=(3,3), activation='relu', input_shape = ( 128, max_len,1)))
model.add(Conv2D(16, 3, strides=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None))
model.add(Conv2D(16, 3, strides=(1,1), activation='relu'))
model.add(Conv2D(16, 3, strides=(1,1), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=None))
model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(len(le.classes_.tolist()),  activation='sigmoid'))

opt = Adadelta()

p_model = multi_gpu_model(model, gpus=3)
p_model.compile(loss='mean_squared_error',optimizer=opt, metrics=['mean_squared_error'])
batch_size = 30
p_model.fit_generator(generator(batch_size), epochs=10,  verbose=1,  shuffle=False, steps_per_epoch=math.ceil((len(files))/batch_size),max_queue_size=1, callbacks = [cb])
p.save(p_model, 'conv_model1')		
			
			
