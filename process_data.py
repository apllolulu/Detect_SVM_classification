"""
https://blog.csdn.net/normol/article/details/84230890
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from glob import glob
from sklearn.model_selection import train_test_split
from random import shuffle
import cv2
import os

def get_id_from_file_path(file_path):
  res_str =  file_path.split(os.path.sep)[-1].replace('.jpg', '').split('_')[-1]
  res_int = int(res_str, 2)


  res = '{:07b}'.format(res_int)
  #print("res:",res)
  return res
    


def data_gen(list_files, train=True):
  shuffle(list_files)
  if train:
    X_train = []
    y_train = []

    for file in list_files:
      X = cv2.imread(file)
      Y = get_id_from_file_path(file)
      

      X_train.append(X)
      y_train.append(Y)

      # 水平翻转
      h_flip = cv2.flip(X, 1)
      X_train.append(h_flip)
      y_train.append(Y)

      # 垂直翻转

      v_flip = cv2.flip(X, 0)
      X_train.append(v_flip)
      y_train.append(Y)

      # 水平垂直翻转
      hv_flip = cv2.flip(X, -1)
      X_train.append(hv_flip)
      y_train.append(Y)


    return np.array(X_train), np.array(y_train)
  else:
    X_test = []
    y_test = []

    for file in list_files:
      X = cv2.imread(file)

      Y = get_id_from_file_path(file)
     

      X_test.append(X)
      y_test.append(Y)
    return np.array(X_test), np.array(y_test)

#train_dir = '../input/train'
#tag = dict([(p,w) for _,p,w in read_csv('../input/train.csv').to_records()]) 

labeled_files = glob('../input/train/*.jpg')
train, val = train_test_split(labeled_files, test_size=0.2, random_state=101010)

X_train, y_train = data_gen(train,train=True)
X_test, y_test = data_gen(val,train=False)

#print('Training data shape: ', X_train.shape)
#print('Training labels shape: ', y_train.shape)
#print('Test data shape: ', X_test.shape)
#print('Test labels shape: ', y_test.shape)



print("y_train:",set(y_train))
print("y_test:",set(y_test))

