
# coding: utf-8

# In[3]:


import sys
import csv
import numpy as np
import random

def Training_GD(x, y, w, n, rate, steps):
    xT = np.transpose(x)
    for i in range(steps):   
        loss = y - np.dot(x, w)        
        G = (-2)  * np.dot(xT, loss) 
        G_normalize = np.sum(G ** 2) / n
        #update the gradient
        w = w - (rate * (G / (G_normalize ** 0.5)))
        
    return w


data = list()
with open('train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    for i,rows in enumerate(reader):
        if i % 18 != 11:
            row = rows
            for i in range(3): 
                row.pop(0)
            row = list(row)
            data += row    
csvfile.close()
#getting data from train data

for i in range(24):
    data.pop(0)
    

data = np.asarray(data, dtype = float)
data = np.reshape(data, (4080, 24))
#remake the data type as ndarray
feature = dict()
for i in range(17):
    feature[i] = list()
#set feature as a dictionary
#PM2.5 data = feature[9]

for i in range(4080):
    for j in range(17):
        if i % 17 == j:
            temp = data[i]
            feature[j] = np.append(feature[j], temp, axis = 0)
#add data to the feature dictionary

train_data = list()
temp = dict()
for i in range(17):
    temp_source = feature[i]
    temp[i] = list()
    for j in range(5750):
        temp[i] = np.append(temp[i], temp_source[j: j + 9], axis = 0)
del temp_source
        
train_data = temp[8]
train_data = np.asarray(train_data)
train_data = np.reshape(train_data, (-1, 9))
for i in range(9, 10):
    temp_array = np.asarray(temp[i])
    temp_array = np.reshape(temp_array, (-1, 9))
    train_data = np.append(train_data, temp_array, axis = 1)
del temp_array
bias_train = np.ones((5750, 1), dtype = float)
train_data = np.append(train_data, bias_train, axis = 1)
#to make the big data matrix

train_data_y = list()
train_data_y = feature[9]
train_data_y = train_data_y[9: -1]
train_data_y = np.asarray(train_data_y)
train_data_y = np.reshape(train_data_y, (-1, 1))

weight = np.random.rand((2 * 9 + 1), 1)
#set the weight scalar
weight = Training_GD(train_data, train_data_y, weight, 5750, 0.00001, 500000)

np.save('hw1.npy', weight)

