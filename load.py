# encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
import collections


# def reformat(samples, labels):
#     #（图片高，图片宽，通道数，图片数） -》 （图片数，图片宽，图片高，通道数）
#     #（   0,    1,    2,      3）  -》 （   3，   0,    1,      2）
#     #new = np.transpose(samples,(3,0,1,2)).astype(np.float32)
#     new=samples.astype(np.float32)
    
#     labels = np.array([x[0] for x in labels])
#     one_hot_labels = []

#     for num in labels:
#         one_hot = [0.0]*6
#         if num == 6:
#               one_hot[0] = 1.0
#         else:
#           one_hot[num] = 1.0
#         one_hot_labels.append(one_hot)
        
#     labels = np.array(one_hot_labels)
#     return new, labels

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

# def normalize(samples):
#     # (R+G+B)/3
#     # 0~255 -> -1.0~1.0
#     a = np.add.reduce(samples, keepdims=True, axis=3)
#     a = a/3.0
#     return a/128.0-1.0

# def distribution(labels, name):
#     arr2=[]
#     for i in range(len(labels)):
#         for j in labels[i]:
#            arr2.append(int(j))
#     result=pd.value_counts(arr2) 
#     plt.figure()
#     result.plot.bar()
#     plt.title(name)
#     plt.show()
#     pass

def inspect(dataset, labels, i):
    print(labels[i])
    plt.imshow(dataset[i].squeeze())
    plt.show()

##运行原始数据
train = scio.loadmat('train1.mat')
test = scio.loadmat('test1.mat')

###运行ToMat产生的数据
# train = scio.loadmat('train.mat')
# test = scio.loadmat('test.mat')

train_samples = train['X']
train_labels = train['y']

test_samples = test['X']
test_labels = test['y']

##运行原始数据需要接触comment
# train_labels= train_labels.astype(np.int)
# train_labels = [[x] for x in train_labels]
# train_labels=np.array(train_labels)
# test_labels= test_labels.astype(np.int)
# test_labels = [[x] for x in test_labels]
# test_labels=np.array(test_labels)

_train_labels = reformat(train_labels)


_test_labels = reformat( test_labels)

# _train_samples = normalize(n_train_samples)
# _test_samples = normalize(n_test_samples)

_train_samples = train_samples
_test_samples = test_samples

print(_train_samples.shape)
print(_test_samples.shape)
print(_train_labels.shape)
print(_test_labels.shape)


num_labels = 6
image_size = 64
image_size1 = 64
num_channels = 1

if __name__ == '__main__':

    inspect(_train_samples, _train_labels,0 )







