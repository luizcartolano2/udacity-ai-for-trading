import numpy as np

def softmax(L):
    expL = np.exp(L)
    return np.divide (expL, expL.sum())

