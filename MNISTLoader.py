import gzip
import pickle as cp
import numpy as np
import random

def loadDataset():
    f = gzip.open('mnist.pkl.gz','rb')

    training_data , validation_data , test_data = cp.load(f , encoding='bytes')
    f.close()
    return(training_data , validation_data , test_data)

def loadWrapper():

    tr_ds , va_ds , te_ds = loadDataset()

    training_inputs = [np.reshape(x,(784,1)) for x in tr_ds[0]]

    validation_inputs = [np.reshape(x,(784,1)) for x in va_ds[0]]

    test_inputs = [np.reshape(x,(784,1)) for x in te_ds[0]]

    training_results = [vectorizedResult(y) for  y in tr_ds[1]]

    validation_results=[vectorizedResult(y) for y in va_ds[1]]

    test_results = [vectorizedResult(y) for y in te_ds[1]]

    training_data = list(zip(training_inputs , training_results))

    validation_data = list(zip(validation_inputs , validation_results))

    test_data = list(zip(test_inputs , test_results))

    return (training_data , validation_data , test_data)

def vectorizedResult(y):
     
    r = np.zeros((10,1))
    r[y] =1.0
    return  r

