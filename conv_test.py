from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
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
def generator(size):
	input_batch = []
	output_batch = []
	for i in range(size):
		file_name = files[i]
		spectrogram = imread("trimmed_curated_spec/" + file_name +".png")
		
		label = labels[i]

		input_batch.append(spectrogram)
		temp = [0]*len(le.classes_.tolist())
		for item in le.transform(label.split(',')):
			temp[item] = 1
		output_batch.append(temp.copy())
	return np.array(input_batch.copy()).reshape(size, 128, max_len, 1), np.array(output_batch.copy())

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
				return np.array(input_batch.copy()).reshape(batch_size, 128, max_len, 1), np.array(output_batch.copy())
				input_batch = []
				output_batch = []
p_model = p.load('conv_model1')
x, y = generator(50)
y_predicted = p_model.predict(x)
count = 0
for i in range(50):
	actual = y[i]
	predicted = y_predicted[i]
	cleaned_pred = [0]*len(le.classes_.tolist())
	for j in range(len(predicted)):
		if predicted[j] > .5:
			cleaned_pred[j] = 1
	if actual.tolist()==cleaned_pred:
		count+=1
	#print("Pred",i," : ", cleaned_pred)
	#print("Actual",i," : ", actual)
print(count/50)

	
			
			
