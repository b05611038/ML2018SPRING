
# coding: utf-8

# In[3]:


import csv
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.externals import joblib

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
    

data = np.asarray(data, dtype = np.float)
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
    for j in range(5757):
        temp[i] = np.append(temp[i], temp_source[j: j + 2], axis = 0)
del temp_source
        
train_data = temp[0]
train_data = np.asarray(train_data)
train_data = np.reshape(train_data, (5757, -1))
for i in range(1, 17):
    temp_array = np.asarray(temp[i])
    temp_array = np.reshape(temp_array, (-1, 2))
    train_data = np.append(train_data, temp_array, axis = 1)
del temp_array
#to make the big data matrix

train_data_y = list()
train_data_y = feature[9]
train_data_y = train_data_y[2: -1]
train_data_y = np.asarray(train_data_y)


model = MLPRegressor(hidden_layer_sizes=(11514, ), learning_rate = 'constant')
model.fit(train_data, train_data_y)

joblib.dump(model, 'hw1_best.pkl', compress = 3)

