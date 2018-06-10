import sys
import csv
import os
import random
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import keras
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Input, Flatten, Embedding, Dot, dot, Lambda, Reshape

print('start to read file and build pandas dataframe')
test_data = pd.read_csv(sys.argv[1])
movie_data = pd.read_fwf(sys.argv[3])
user_data = pd.read_csv(sys.argv[4])
print('the process done')

print('start to load model')
model = load_model('mf_model.h5')
print('model loading done\nstart prediction')

pre_user = np.asarray(test_data.iloc[:, 1]).reshape(test_data.shape[0], 1)
pre_movie = np.asarray(test_data.iloc[:, 2]).reshape(test_data.shape[0], 1)

prediction = model.predict([pre_user, pre_movie], batch_size = 500, verbose = 1)

for i in range(len(prediction)):
    if prediction[i, 0] > 5:
        prediction[i, 0] = 5


print('prediction done\nstart file output')
outfile = open(sys.argv[2], 'w')
lines = ['TestDataID,Rating\n']

for i in range(len(prediction)):
    lines.append(str(i + 1) + ',' + str(float(prediction[i, 0])) + '\n')

outfile.writelines(lines)
outfile.close()
print('all done')
