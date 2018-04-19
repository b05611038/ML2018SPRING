
# coding: utf-8

# In[ ]:


import sys
import os
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.backend.tensorflow_backend import set_session
from keras.utils import np_utils
from keras.models import Sequential, load_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
set_session(tf.Session(config = config))

model = load_model('model_15(vgglike1.3)nG.h5')

Test = pd.read_csv(sys.argv[1])

test = np.asarray([])
for i in range(Test.shape[0]):
	test = np.concatenate((test, np.fromstring(Test.iloc[i][1], dtype = np.float, sep = ' ')), axis = 0)

test_x = test / 255
test_x = np.reshape(test_x, (Test.shape[0], 48, 48, 1)).astype('float32')

prediction = model.predict_classes(x = test_x, batch_size = 1000, verbose = 0)

outfile = open(sys.argv[2], 'w')
lines = ['id,label\n']

for i in range(len(prediction)):
	lines.append(str(i) + ',' + str(prediction[i]) + '\n')
outfile.writelines(lines)
outfile.close

