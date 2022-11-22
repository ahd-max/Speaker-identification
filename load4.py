# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 20:37:12 2022

@author: Hangdong AN
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
import collections


def reformat(labels):
    labels = np.array([x[0] for x in labels.transpose()])  # slow code, whatever
    # print(labels)
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0]*6
        if num == 6:
           one_hot[0] = 1.0
        else:
              one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return labels

##运行原始数据
train = scio.loadmat('train4.mat')
test = scio.loadmat('test4.mat')



train_samples = train['X']
train_labels = train['y']

test_samples = test['X']
test_labels = test['y']

_train_labels = reformat(train_labels)


_test_labels = reformat( test_labels)


_train_samples = train_samples
_test_samples = test_samples

print(_train_samples.shape)
print(_test_samples.shape)
print(_train_labels.shape)
print(_test_labels.shape)
def inspect(dataset, labels, i):
    print(labels[i])
    plt.imshow(dataset[i].squeeze())
    plt.show()

num_labels = 6
image_size = 64
image_size1 = 64
num_channels = 1

if __name__ == '__main__':
    
    
    inspect(_train_samples, _train_labels, 1)
