import numpy as np
from random import shuffle


def svm_loss_vectorized(W, X, y, reg):

  loss = 0.0
  dW = np.zeros(W.shape) 
  scores = X.dot(W)
  num_train = X.shape[0]

  #利用np.arange(),correct_class_score变成了 (num_train,y)的矩阵
  correct_class_score = scores[np.arange(num_train),y]
  correct_class_score = np.reshape(correct_class_score,(num_train,-1))
  margins = scores - correct_class_score + 1
  margins = np.maximum(0, margins)
  #然后这里计算了j=y[i]的情形，所以把他们置为0
  margins[np.arange(num_train),y] = 0
  loss += np.sum(margins) / num_train
  loss += reg * np.sum( W * W)

  margins[margins > 0] = 1
  #因为j=y[i]的那一个元素的grad要计算 >0 的那些次数次
  row_sum = np.sum(margins,axis=1)
  margins[np.arange(num_train),y] = -row_sum.T
  #把公式1和2合到一起计算了
  dW = np.dot(X.T,margins)
  dW /= num_train
  dW += reg * W

  return loss, dW
