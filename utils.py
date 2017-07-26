import numpy as np
import math

def relu(x):
    xx = x.copy()
    xx[xx<0] = 0
    #return 1 / (1 + np.exp(-x))
    return xx

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    v = e_x / e_x.sum(axis=1)[:, None]
    return v

def softmax_loss(x, y):
    res = 0
    for i in range(0, len(y)):
        res += -math.log(x[i, y[i]] + 1e-15)
    return res / len(y)

