import time
import random
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from glob import glob
from sklearn.model_selection import train_test_split
from random import shuffle
import cv2
import os
from sklearn.linear_model import LogisticRegression
#from sklearn import svm
from sklearn.externals import joblib
from cs231n.classifiers import LinearSVM
import pandas as pd



def get_id_from_file_path(file_path):
	res_str =  file_path.split(os.path.sep)[-1].replace('.jpg', '').split('_')[-1]
	res_int = int(res_str, 2)
	res_int = '{:07b}'.format(res_int)

	if res_str == '0100010':
		res = 0
	elif res_str == '0111010':
		res = 1
	elif res_str == '0001010':
		res = 2
	elif res_str == '1111111':
		res = 3
	return res_int,res


def get_predict_from_model(y_val_pred):
	encoder = []
	for pred in y_val_pred:
		if pred == 0:
			encode = '0100010'
		elif pred == 1:
			encode = '0111010'
		elif pred == 2:
			encode = '0001010'
		elif pred == 3:
			encode = '1111111'
		else:
			encode = '1111111'

		encode = int(encode, 2)
		encode = '{:07b}'.format(encode)
		encoder.append(encode)

	return encoder


def data_gen(list_files, batch_size,train=True):

	shuffle(list_files)

	if train:
		X_train = []
		y_train = []


		for file in list_files:
			X = cv2.imread(file)
			X=cv2.resize(X,(64,32),interpolation=cv2.INTER_CUBIC)# INTER_AREA
			X = np.array(X)

			Y = get_id_from_file_path(file)[1]

			X_train.append(X)
			y_train.append(Y)
			"""
			# 水平翻转
			h_flip = cv2.flip(X, 1)

			h_flip = np.array(h_flip)

			X_train.append(h_flip)
			y_train.append(Y)

			# 垂直翻转

			v_flip = cv2.flip(X, 0)
			v_flip = np.array(v_flip)

			X_train.append(v_flip)
			y_train.append(Y)

			# 水平垂直翻转

			hv_flip = cv2.flip(X, -1)
			hv_flip = np.array(hv_flip)

			X_train.append(hv_flip)
			y_train.append(Y)
		    """
		#	yield np.array(X_train), np.array(y_train)
		#	X_train = []
		#	y_train = []
            
		return np.array(X_train), np.array(y_train)
	else:

		X_val = []
		y_val = []

		for file in list_files:
			X = cv2.imread(file)
			X=cv2.resize(X,(64,32),interpolation=cv2.INTER_CUBIC)# INTER_AREA

			#X = np.array(X)
			Y = get_id_from_file_path(file)[1]

			X_val.append(X)
			y_val.append(Y)

		#	yield np.array(X_val), np.array(y_val)
		#	X_val = []
		#	y_val = []
			
		return np.array(X_val), np.array(y_val)



#train_dir = '../input/train'
#tag = dict([(p,w) for _,p,w in read_csv('../input/train.csv').to_records()]) 

labeled_files = glob('../input/train/*.jpg')
train, val = train_test_split(labeled_files, test_size=0.2, random_state=101010)

"""
get_batches_train = data_gen(train,batch_size,train=True)
X_train, y_train=next(get_batches_train)

get_batches_val = data_gen(val,batch_size,train=False)
X_val, y_val = next(get_batches_val)
"""

batch_size = 1
X_train, y_train = data_gen(train,batch_size,train=True)
X_val, y_val = data_gen(val,batch_size,train=False)

#print('Test data shape: ', X_val.shape)
#print('Test labels shape: ', y_val.shape)

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))

#print('Training data shape: ', X_train.shape)
#print('Training labels shape: ', y_train.shape)


X_train=np.array(X_train,dtype=np.float) 
X_val=np.array(X_val,dtype=np.float)

mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image

X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

# 模型训练
tic = time.time()
#clf = svm.SVC()
#clf.fit(X_train, y_train)
svm = LinearSVM()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
                      num_iters=2000, verbose=True)
toc = time.time()
print('That took %fs' % (toc - tic))

plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()


save_path_name="svm_"+"train_model.m"
joblib.dump(svm, save_path_name)


# 模型测试
svm = joblib.load(save_path_name)

#y_train_pred=clf.predict(X_train)
y_train_pred = svm.predict(X_train)
print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))

#y_val_pred=clf.predict(X_val)
y_val_pred = svm.predict(X_val)
print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))

encoder = get_predict_from_model(y_val_pred)
encoder_true = get_predict_from_model(y_val)

output = pd.DataFrame({'label': encoder_true, 'pred': encoder})
output.to_csv('test.csv', index=False)

print(output.head())