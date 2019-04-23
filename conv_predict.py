import ezPickle as p
from scipy.misc import imread
import pandas as pd
import glob, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
p_model = p.load('conv_model')
le = p.load('le')
names = []
cols = ['fname'] + le.classes_.tolist()
df = pd.DataFrame(columns = cols)
count = 0
for file_name in glob.glob('test_spec/*'):
	spec = imread(file_name).reshape(1,128,1800,1)
	predicted = p_model.predict(spec)[0]
	cleaned_pred = [0]*len(le.classes_.tolist())
	for j in range(len(predicted)):
		if predicted[j] > .5:
			cleaned_pred[j] = 1
	pred_names = le.inverse_transform(cleaned_pred)
	#names.append(file_name[file_name.index('/')+1:-3]+'.wav')
	temp_list = ([file_name[file_name.index('/')+1:-4]] + cleaned_pred)
	print(temp_list)
	print(len(temp_list))
	df.loc[count] = temp_list.copy()
	count+=1
df.to_csv('submission.csv',index=None)
	

		
	
	

