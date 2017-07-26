import numpy as np
from sklearn import datasets
from sklearn.metrics import *
import math

digits = datasets.load_digits()

X = digits['data'][:-100]
Y = digits['target'][:-100]
X_te = digits['data'][-100:]
Y_te = digits['target'][-100:]
Y_dummy = np.zeros((len(X), 10))
for i in range(0, len(X)):
    Y_dummy[i, Y[i]] = 1

print X.shape

# top mlp
# one hidden layer with relu
# one output layer with softmax

hiddensize = 20
w1 = np.random.random((X.shape[1], hiddensize)) - 0.5
w2 = np.random.random((hiddensize, 10)) - 0.5


print w1.shape
print w2.shape

def relu(x):
    xx = x.copy()
    xx[xx<0] = 0
    #return 1 / (1 + np.exp(-x))
    return xx

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    v = e_x / e_x.sum(axis=1)[:, None]
    return v


def loss(x, y):
    res = 0
    for i in range(0, len(y)):
        res += -math.log(x[i, y[i]] + 1e-15)
    return res / len(y)

def predict(x):
    a = x_batch.dot(w1)
    z = relu(a)
    a2 = z.dot(w2)
    pre_batch = softmax(a2)
    return pre_batch

regu = 5

#update
for iter in range(0,100):
    batchsize = len(X)
    for i in range(0, X.shape[0], batchsize):
        print '---'
        x_batch = X[i:i+batchsize, :]
        y_batch = Y[i:i+batchsize]
        y_dummy_batch = Y_dummy[i:i+batchsize, :]
        #feed forward
        a = x_batch.dot(w1)
        z = relu(a)
        #z = np.random.random(z.shape)
        a2 = z.dot(w2)
        pre_batch = softmax(a2)
        batch_loss = loss(pre_batch, y_batch)
        print 'train loss:' + str(batch_loss) + ' eval loss:' + str(loss(predict(X_te), Y_te))
        #back prop
        #back prop w2
        #loss for L / o
        grad_a2 = (pre_batch - y_dummy_batch) / batchsize
        grad_w2 = (z.T.dot(grad_a2) + regu * w2)
        grad_z = grad_a2.dot(w2.T)
        print grad_z.shape
        print x_batch.shape
        #grad_z_a = z * (z - 1)
        grad_z_a = z.copy()
        grad_z_a[grad_z_a > 0] = 1
        grad_z_a[grad_z_a != 1] = 0
        #print grad_z_a
        grad_w1 = (x_batch.T.dot(grad_z * grad_z_a) + regu * w1) 
        w2 -= 0.01 * grad_w2
        w1 -= 0.01 * grad_w1

