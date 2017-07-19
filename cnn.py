
# coding: utf-8

# In[8]:

from sklearn.datasets import load_digits
import numpy as np
data = load_digits()


# In[24]:

test = np.array([1,3,4,5])
test += test
print test


# In[10]:

data['images'][0]


# In[28]:

# 3 * 3 conv
convsize = 3
convnum = 5
# total five conv
w = []
for i in range(0, convnum):
    w.append(np.random.random((convsize, convsize)) - 0.5)

hiddensize = (data['images'][0].shape[0] - convsize + 1) * (data['images'][0].shape[1] - convsize + 1)
#feed forward
print hiddensize
w2 = np.random.random((hiddensize, 10)) - 0.5

def relu(x):
    xx = x.copy()
    xx[xx < 0] = 0
    return xx

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    v = e_x / e_x.sum(axis=1)[:, None]
    return v

def feed(x):
    #conv
    hid = None
    for i in range(0, len(w)):
        mid = []
        for xl in range(0, x.shape[0] - convsize + 1):
            for yl in range(0, x.shape[1] - convsize + 1):
                mid.append(np.sum(x[xl:xl+convsize, yl:yl+convsize] * w[i]))
        print np.array(mid).shape
        if i == 0:
            hid = np.array(mid)
        else:
            hid += np.array(mid)
    print hid.shape
    hid /= hiddensize
    print hid.shape
    z = relu(hid)
    print z.shape
    print w2.shape
    return softmax(z.dot(w2))
    
feed(data['images'][0])

