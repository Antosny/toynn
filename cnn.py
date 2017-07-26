from sklearn.datasets import load_digits
import numpy as np
from utils import *
import math
convweight = []
krow = 3
kcol = 2
for i in range(0, 5):
    convweight.append(np.random.random(krow * kcol) - 0.5)
fcweight = np.random.random((42, 10)) - 0.5

digits = load_digits()

X = digits['images'][:-100]
Y = digits['target'][:-100]
X_te = digits['images'][-100:]
Y_te = digits['target'][-100:]
Y_dummy = np.zeros((len(X), 10))
for i in range(0, len(X)):
    Y_dummy[i, Y[i]] = 1




def conv_1d_to_2d(data, row, column):
    if len(data) != row * column:
        return
    res = np.zeros((row, column))
    for i in range(0, row):
        for j in range(0, column):
            res[i][j] = data[i*column + j]
    return res


def conv_2d_to_matrix(data, kernelrow, kernelcol):
    mid = []
    for xl in range(0, data.shape[0] - kernelrow + 1):
        for yl in range(0, data.shape[1] - kernelcol + 1):
            mid.append(data[xl:xl+kernelrow, yl:yl+kernelcol].flatten())
    return np.array(mid)

def conv_2d_to_matrix_batch(data, kernelrow, kernelcol):
    mid = []
    for single in data:
        mid.append(conv_2d_to_matrix(single, kernelrow, kernelcol))
    return np.array(mid)

def recover_matrix_to_2d(data, row, col):
    if (len(data) != row * col):
        return
    res = np.zeros((row, col))
    idx = 0
    for i in range(0, row):
        for j in range(0, col):
            res[i, j] = data[idx]
            idx += 1
    return res

def recover_matrix_to_2d_batch(data, row, col):
    res = []
    for d in data:
        res.append(recover_matrix_to_2d(d))
    return np.array(res)


#2d matrix with kernellist and recover
def conv(data, kernellist):
    res = None
    resrow = data.shape[0] - krow + 1
    rescol = data.shape[1] - kcol + 1
    for kernel in kernellist:
        tmp = np.dot(conv_2d_to_matrix(data, krow, kcol), kernel)
        if res is None:
            res = tmp
        else:
            res += tmp
    return recover_matrix_to_2d(res, resrow, rescol)

def conv_batch(data, kernellist):
    res = []
    for d in data:
        res.append(conv(d, kernellist))
    return np.array(res)
    

def forward(x):
    #conv, relu
    tmpmtx = relu(conv_batch(x, convweight).reshape(len(x), 42))
    #fc
    return softmax(np.dot(tmpmtx, fcweight))


    
for iter in range(0,1):
    batchsize = len(X)
    for i in range(0, X.shape[0], batchsize):
        x_batch = X[i:i+batchsize, :]
        y_batch = Y[i:i+batchsize]
        y_dummy_batch = Y_dummy[i:i+batchsize, :]
        #forward
        #conv + relu
        x_conv = conv_batch(x_batch, convweight).reshape(len(x_batch), 42)
        x_relu = relu(x_conv)
        #fc
        x_fc = x_relu.dot(fcweight)
        #out
        pre_batch = softmax(x_fc)
        batch_loss = softmax_loss(pre_batch, y_batch)
        print 'train loss:' + str(batch_loss) + ' eval loss:' + str(softmax_loss(forward(X_te), Y_te))





