
# coding: utf-8

# In[ ]:


import sys
import csv
import time
import pandas as pd
import numpy as np
import math
import random

def test_fit(test, weight):
    sequence = np.asarray([])
    one = np.ones(1)
    zero = np.zeros(1)
    for i in range(test.shape[0]):
        z = - np.dot(test[i, :], weight)
        
        if z >= 3:
            f = 1 / (1 + math.exp(3))
        elif z <= -5:
            f = 1 / (1 + math.exp(-5))
        else:
            f = 1 / (1 + math.exp(z))
            
        
        if f >= 0.5:
            sequence = np.append(sequence, one, axis = 0)
        elif f < 0.5:
            sequence = np.append(sequence, zero, axis = 0)
        
    return sequence


def transition(matrix):
    mean = np.sum(matrix, axis = 0) / matrix.shape[0]
    for i in range(len(mean)):
        if mean[i] <= 1:
            mean[i] = 0
            
    mean_matrix = np.asarray([])
    for i in range(matrix.shape[0]):
        mean_matrix = np.append(mean_matrix, mean, axis = 0)
        
    mean_matrix = mean_matrix.reshape(matrix.shape[0], -1)
    var = np.sum(((matrix - mean_matrix) ** 2), axis = 0)
    var = var ** (1 / 2)
    for i in range(len(mean)):
        if mean[i] == 0:
            var = 1      
    
    new_matrix = (matrix - mean_matrix) / var
    #standard score
    
    new_mean = np.sum((new_matrix ** 2), axis = 0)
    new_mean = new_mean ** (1 / 2)
    sequence = new_matrix / new_mean
    #to let data parameter smaller
    sequence = np.nan_to_num(sequence)
    
    return sequence

test = pd.DataFrame(pd.read_csv(sys.argv[5]))
test = test.reset_index().values
test = np.asarray(test, dtype = np.float)
test = np.delete(test, 0, axis = 1)
test = transition(test)

weight = np.load('hw2.npy')

ans = test_fit(test, weight)

file = open(sys.argv[6], 'w')
lines = ['id,label\n']
for i in range(len(ans)):
    lines.append(str(i + 1) + ',' + str(int(ans[i])) + '\n')
file.writelines(lines)
file.close()

