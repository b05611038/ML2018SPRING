
# coding: utf-8

# In[ ]:


import sys
import math
import csv
import numpy as np
import pandas as pd

def standard_score_region_transition(matrix):
    i = 0
    
    while True:
        if i >= matrix.shape[1]:
            break
            
        sum_filter = np.sum(matrix, axis = 0)    
        if sum_filter[i] > matrix.shape[0]:
            mean = sum_filter[i] / matrix.shape[0]
            
            var = 0
            for j in range(matrix.shape[0]):
                var = var + ((matrix[j][i] - mean) ** 2)
            var = var / matrix.shape[0]
            var = var ** (1 / 2)
                      
            standard_point = []
            for j in range(matrix.shape[0]):
                standard_point.append((matrix[j][i] - mean) / var)
                
            insert_matrix = np.asarray([])
            for j in range(matrix.shape[0]):
                if standard_point[j] > 2:
                    insert_matrix = np.append(insert_matrix, [0, 0, 0, 0, 0, 0, 1], axis = 0)
                elif standard_point[j] <= 2 and standard_point[j] > 1:
                    insert_matrix = np.append(insert_matrix, [0, 0, 0, 0, 0, 1, 0], axis = 0)
                elif standard_point[j] <= 1 and standard_point[j] > 0:
                    insert_matrix = np.append(insert_matrix, [0, 0, 0, 0, 1, 0, 0], axis = 0)
                elif standard_point[j] == 0:
                    insert_matrix = np.append(insert_matrix, [0, 0, 0, 1, 0, 0, 0], axis = 0)
                elif standard_point[j] < 0 and standard_point[j] >= -1:
                    insert_matrix = np.append(insert_matrix, [0, 0, 1, 0, 0, 0, 0], axis = 0)
                elif standard_point[j] < -1 and standard_point[j] >= -2:
                    insert_matrix = np.append(insert_matrix, [0, 1, 0, 0, 0, 0, 0], axis = 0)
                elif standard_point[j] < -2:
                    insert_matrix = np.append(insert_matrix, [1, 0, 0, 0, 0, 0, 0], axis = 0)
            insert_matrix = insert_matrix.reshape(-1, 7)
            
            matrix = np.delete(matrix, i, axis = 1)
            temp_left = matrix[0 : matrix.shape[0], 0 : (i - 1)]
            temp_right = matrix[0 : matrix.shape[0], i :]
            matrix = np.concatenate((temp_left, insert_matrix, temp_right), axis = 1)
            i = i + 1
            
        elif sum_filter[i] < matrix.shape[0]:
            i = i + 1
    
    matrix = np.nan_to_num(matrix)
    
    return matrix

data = pd.DataFrame(pd.read_csv(sys.argv[3]))
data = data.reset_index().values
data = np.asarray(data, dtype = np.float)
data = np.delete(data, 0, axis = 1)
data = standard_score_region_transition(data)

train_y = pd.DataFrame(pd.read_csv(sys.argv[4]))
train_y = train_y.reset_index().values
train_y = np.asarray(train_y, dtype = np.float)
train_y = np.delete(train_y, 0, axis = 1)
loss_y = np.zeros(1)
train_y = np.transpose(train_y)
train_y = np.append(loss_y, train_y)
train_y = np.reshape(train_y, (-1, 1))

test = pd.DataFrame(pd.read_csv(sys.argv[5]))
test = test.reset_index().values
test = np.asarray(test, dtype = np.float)
test = np.delete(test, 0, axis = 1)
test = standard_score_region_transition(test)

pro = np.zeros(data.shape[1])
pro_no = np.zeros(data.shape[1])
for i in range(data.shape[1]):
    for j in range(data.shape[0]):
        if data[j][i] == 1 and train_y[j][0] == 1:
            pro[i] = pro[i] + 1
        if data[j][i] == 1 and train_y[j][0] == 0:
            pro_no[i] = pro_no[i] + 1
            
pro = pro / (pro_div + pro)
pro = np.nan_to_num(pro)
pro = np.reshape(pro, (-1, 1))

pro_pre = np.dot(test, pro)
pro_pre = pro_pre / 14
#feature numbers

sequence = np.asarray([])
one = np.ones(1)
zero = np.zeros(1)
for i in range(test.shape[0]):
    if pro_pre[i] >= 0.5:
        sequence = np.append(sequence, one, axis = 0)
    elif pro_pre[i] < 0.5:
        sequence = np.append(sequence, zero, axis = 0)

file = open(sys.argv[6], 'w')
lines = ['id,label\n']
for i in range(len(sequence)):
    lines.append(str(i + 1) + ',' + str(int(sequence[i])) + '\n')
file.writelines(lines)
file.close()

