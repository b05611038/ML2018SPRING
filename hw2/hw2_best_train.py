
# coding: utf-8

# In[1]:


import sys
import csv
import time
import pandas as pd
import numpy as np
import math
import random

start_time = time.time()
def GradientDescent_parameters(train_data, train_data_y, weight, lr, a, steps):
    s_gra = np.random.rand(len(train_data[0]))
    previous_gra = np.zeros(len(train_data[0]))

    for i in range(steps):
        num = random.randint(0, train_data.shape[0] - 1)
        x = train_data[num, :]
        x_t = np.transpose(x)
        
        hypo = - np.dot(x, weight)
        hypo = 1 / (1 + math.exp(hypo))
        
        loss = hypo - train_data_y[num][0]  
        gra = loss * x_t + a * previous_gra
        previous_gra = gra
        s_gra += gra ** 2
        ada = np.sqrt(s_gra)
        weight = weight - lr * gra/ada
        
    return weight

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


data = pd.DataFrame(pd.read_csv(sys.argv[3]))
data = data.reset_index().values
data = np.asarray(data, dtype = np.float)
data = np.delete(data, 0, axis = 1)
data = transition(data)

train_y = pd.DataFrame(pd.read_csv(sys.argv[4]))
train_y = train_y.reset_index().values
train_y = np.asarray(train_y, dtype = np.float)
train_y = np.delete(train_y, 0, axis = 1)
loss_y = np.zeros(1)
train_y = np.transpose(train_y)
train_y = np.append(loss_y, train_y)
train_y = np.reshape(train_y, (-1, 1))

weight = np.random.rand(len(data[0]))
weight = GradientDescent_parameters(data, train_y, weight, 10, 0.4, 10000000)

np.save('hw2.npy', weight)

