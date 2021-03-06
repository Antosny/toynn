# coding: utf-8


from utils import *
from sklearn.datasets import load_digits
import numpy as np
hidden = 5

digits = load_digits()

X = digits['images'][:-100]
Y = digits['target'][:-100]
X_te = digits['images'][-100:]
Y_te = digits['target'][-100:]
Y_dummy = np.zeros((len(X), 10))
for i in range(0, len(X)):
    Y_dummy[i, Y[i]] = 1

w1 = (np.random.random((X[0].shape[0], hidden)) - 0.5)
w2 = (np.random.random((hidden, hidden)) - 0.5)
#w2 = np.identity(hidden)
w3 = (np.random.random((hidden, 10)) - 0.5)

#final output label
def forward(x):
    preh = None
    a = x.T.dot(w1)
    print a.shape
    for data in a:
        if preh is None:
            h = tanh(data)
        else:
            h = tanh(data + preh.dot(hidden))
        preh = h
    return softmax([preh.dot(w3)])

lr = 0.1

predhlist = []

#for j in range(0, len(X)):
#    x = X[j]
#    y = Y[j]
#    y_du = Y_dummy[j]
#    a = x.T.dot(w1)
#    preh = None
#    hlist = []
#    for data in a:
#        if preh is None:
#            h = tanh(data)
#        else:
#            h = tanh(data + preh.dot(hidden))
#        preh = h
#        hlist.append(h.reshape(1, hidden))
#    predhlist.append(hlist[-1].reshape(hidden))#

#predarray = np.array(predhlist)
#predarray = np.random.random(predarray.shape) - 0.5

#for iter in range(0, 200):
#    #print predarray[0]
#    pre_batch = softmax(predarray.dot(w3))
#    #print pre_batch.shape
#    batch_loss = softmax_loss(pre_batch, Y)
#    print 'loss:' + str(batch_loss)
#    grada2 = (pre_batch - Y_dummy) / len(X)
#    gradw2 = predarray.T.dot(grada2)
#    w3 -= lr * gradw2


#     pred = []
#     w3grad = []
#     for i in range(0, len(X)):
#         p = softmax([predhlist[i].dot(w3).reshape(10)])
#         pred.append(p.reshape(10))
#         gradsoft = p - Y_dummy[i]
#         w3grad.append(predhlist[i].T.dot(gradsoft))
#         #w3 -= lr * w3grad[-1]
#     print softmax_loss(np.array(pred), Y)
#     w3 -= lr * np.mean(w3grad, axis=0)

for iter in range(0,20):
    batchsize = len(X)
    print 'epoch:' + str(iter)
    print 'lenpredh:' + str(len(predhlist))
    for i in range(0, X.shape[0], batchsize):
        x_batch = X[i:i+batchsize, :]
        y_batch = Y[i:i+batchsize]
        y_dummy_batch = Y_dummy[i:i+batchsize, :]
        #batch
        w1grad = []
        w2grad = []
        w3grad = []
        pred = []
        for j in range(0, len(x_batch)):
            x = x_batch[j]
            y = y_batch[j]
            y_du = y_dummy_batch[j]
            #forward
            a = x.T.dot(w1)
            hlist = []
            gradhlist = []
            gradh = [] #need to reverse in later
            preh = None
            for data in a:
                if preh is None:
                    h = tanh(data)
                else:
                    h = tanh(data + preh.dot(hidden))
                preh = h
                gradtanh = 1 - preh * preh
                #gradtanh[gradtanh > 0] = 1
                #gradtanh[gradtanh != 1] = 0
                gradhlist.append(gradtanh)
                hlist.append(h.reshape(1, hidden))
            if iter == 0:
                predhlist.append(hlist[-1])
            p = softmax([hlist[-1].dot(w3).reshape(10)])
            if iter > 0:
                p = softmax(predhlist[j].dot(w3))
            pred.append(p.reshape(10))
            gradsoft = p - y_du
            for l in range(len(hlist) - 1, -1, -1):
                #output
                corridx = len(hlist) - l - 1
                if l == len(hlist) - 1:
                    gradh.append(gradsoft.dot(w3.T).reshape(1, hidden))
                else:
                    gradtanh = gradhlist[corridx]
                    gradh.append((gradtanh * gradh[-1]).dot(w2.T))
            gradh.reverse()
            gradw3 = predhlist[j].T.dot(gradsoft)
            w3grad.append(gradw3)
            #cal grad w2, gradw2 = gradh2 * h2
            tempw2 = []
            for i in range(0, len(hlist)):
                gradtanh = gradhlist[corridx]
                tempw2.append(hlist[i].T.dot(gradh[i] * gradtanh))
            w2grad.append(np.array(tempw2).mean(axis=0))
            print '---'
            print tempw2[0]
            print tempw2[-1]
            print '---'
            #cal grad w1
            tempw1 = []
            for i in range(0, len(hlist)):
                gradtanh = gradhlist[corridx]
                tempw1.append(x[i].reshape(8, 1).dot(gradh[i] * gradtanh))
            w1grad.append(np.array(tempw1).mean(axis=0))
        print softmax_loss(np.array(pred), y_batch)
        print np.mean(w2grad, axis=0)
        #w1 -= lr * np.mean(w1grad, axis=0)
        w2 -= lr * np.mean(w2grad, axis=0)
        w3 -= lr * np.mean(w3grad, axis=0)


# In[ ]:



