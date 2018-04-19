
# coding: utf-8

# In[ ]:


import sys
import os
import csv
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
set_session(tf.Session( config = config))

Data = pd.read_csv(sys.argv[1])
data_temp = {}
for i in range(28):
	data_temp[i] = np.asarray([])
	for j in range(1000):
		data_temp[i] = np.concatenate((data_temp[i], np.fromstring(Data.iloc[1000 * i + j][1], dtype = np.float, sep=' ')), axis = 0)

data_temp[28] = np.asarray([])
for i in range(Data.shape[0] - 28000):
	data_temp[28] = np.concatenate((data_temp[28], np.fromstring(Data.iloc[i + 28000][1], dtype = np.float, sep = ' ')), axis = 0)

data = np.asarray([])
for i in range(29):
	data = np.concatenate((data, data_temp[i]), axis = 0)

del data_temp
	
train_x = np.reshape(data, (Data.shape[0], 48, 48, 1)).astype('float32')

del data

train_x = train_x / 255

train_y = np.asarray([])
for i in range(Data.shape[0]):
    if Data.iloc[i][0] == 0:
        train_y = np.append(train_y, np.asarray([1, 0, 0, 0, 0, 0, 0]))
    elif Data.iloc[i][0] == 1:
        train_y = np.append(train_y, np.asarray([0, 1, 0, 0, 0, 0, 0]))
    elif Data.iloc[i][0] == 2:
        train_y = np.append(train_y, np.asarray([0, 0, 1, 0, 0, 0, 0]))
    elif Data.iloc[i][0] == 3:
        train_y = np.append(train_y, np.asarray([0, 0, 0, 1, 0, 0, 0]))
    elif Data.iloc[i][0] == 4:
        train_y = np.append(train_y, np.asarray([0, 0, 0, 0, 1, 0, 0]))
    elif Data.iloc[i][0] == 5:
        train_y = np.append(train_y, np.asarray([0, 0, 0, 0, 0, 1, 0]))
    elif Data.iloc[i][0] == 6:
        train_y = np.append(train_y, np.asarray([0, 0, 0, 0, 0, 0, 1]))
    else:
        train_y = np.append(train_y, np.asarray([0, 0, 0, 0, 0, 0, 0]))

train_y = np.reshape(train_y, (-1, 7))

del Data

model = Sequential()

model.add(Conv2D(filters = 64,
	             kernel_size = (3, 3),
	             padding = 'same',
	             input_shape = (48, 48, 1),
	             activation = 'relu'))

for i in range(1):
	model.add(Conv2D(filters = 64,
	                 kernel_size = (3, 3),
	                 padding = 'same',
	                 activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

for i in range(3):
	model.add(Conv2D(filters = 128,
                         kernel_size = (3, 3),
                         padding = 'same',
                         activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

for i in range(3):
	model.add(Conv2D(filters = 256,
	                 kernel_size = (3, 3),
			 padding = 'same',
                         activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

for i in range(2):
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.5))
    
model.add(Dense(7, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

print(model.summary())

gen = ImageDataGenerator(featurewise_center = False,
                         samplewise_center = False,
                         rotation_range = 8,
                         width_shift_range = 0.08,
                         shear_range = 0.12,
                         height_shift_range = 0.08,
                         zoom_range = 0.1,
                         data_format = 'channels_last')

gen.fit(train_x)
train_generator = gen.flow(train_x, train_y, batch_size = 1500)

model.fit_generator(train_generator,
                    steps_per_epoch = 600,
                    epochs = 60,
                    verbose = 1,
                    max_q_size = 10)

model.save('model_15(vgglike1.3)nG.h5') 

