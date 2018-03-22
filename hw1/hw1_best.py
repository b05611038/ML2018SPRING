
# coding: utf-8

# In[ ]:


import sys
import csv
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.externals import joblib

test = list()
with open(sys.argv[1], 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    for i,rows in enumerate(reader):
        if i % 18 != 10:
            row = rows
            for i in range(9): 
                row.pop(0)
            row = list(row)
            test += row    
csvfile.close()

test = np.asarray(test, dtype = float)
test = np.reshape(test, (260, -1))
#make the test data matrix

model = joblib.load('hw1_best.pkl')

prediction = module.predict(test)

prediction = list(prediction)
file = open(sys.argv[2], 'w')

lines = ['id,value\n']
for i in range(len(prediction)):
    lines.append('id_' + str(i) + ',' + str(prediction[i]) + '\n')
file.writelines(lines)
file.close()

