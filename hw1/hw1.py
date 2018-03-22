
# coding: utf-8

# In[ ]:


import sys
import csv
import random
import numpy as np

test = list()
with open(sys.argv[1], 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    for i,rows in enumerate(reader):
        if i % 18 == 8 or i % 18 == 9:
            row = rows
            for i in range(2): 
                row.pop(0)
            row = list(row)
            test += row    
csvfile.close()

test = np.asarray(test, dtype = float)
test = np.reshape(test, (-1, 18))
bias_test = np.ones((260, 1), dtype = float)
test = np.append(test, bias_test, axis = 1)
#make the test data matrix

weight = np.load('hw1.npy')

prediction = np.dot(test, weight)
prediction = list(prediction)

file = open(sys.argv[2], 'w')

lines = ['id,value\n']
for i in range(len(prediction)):
    lines.append('id_' + str(i) + ',' + str(prediction[i][0]) + '\n')
file.writelines(lines)
file.close()

