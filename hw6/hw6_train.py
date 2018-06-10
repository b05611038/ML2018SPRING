import sys
import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import keras
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Input, Flatten, Embedding, Dot, Add, Reshape

def model_mf(user_input, movie_input, attribute, user_num, movie_num):
        u = Embedding(user_num, attribute, input_length = 1)(user_input)
        u = Dropout(0.5)(u)
        u = Reshape((1, -1))(u)

        u_bias = Embedding(user_num, 1, input_length = 1)(user_input)
        u_bias = Dropout(0.5)(u_bias)
        u_bias = Flatten()(u_bias)

        m = Embedding(movie_num, attribute, input_length = 1)(movie_input)
        m = Dropout(0.5)(m)
        m = Reshape((1, -1))(m)

        m_bias = Embedding(movie_num, 1, input_length = 1)(movie_input)
        m_bias = Dropout(0.5)(m_bias)
        m_bias = Flatten()(m_bias)
       
        x = Dot(axes = -1)([u, m])
        x = Add()([u_bias, m_bias, x])
        x = Flatten()(x)
        x = Dense(1, activation = 'relu')(x)

        model = Model([user_input, movie_input], x)

        return model


attribute = 128
#set the parameter

print('start to read file and build pandas dataframe')
train_data = pd.read_csv('train.csv')
print('the process done')

print('start to build the dictionary of users and movies')
user_id = {}
user_count = 0
for i in range(train_data.shape[0]):
    if train_data.iloc[i, 1] not in user_id:
        user_id[user_count] = train_data.iloc[i, 1]
        user_count += 1
print('user dictionary building done')

movie_id = {}
movie_count = 0
for i in range(train_data.shape[0]):
    if train_data.iloc[i, 2] not in movie_id:
        movie_id[movie_count] = train_data.iloc[i, 2]
        movie_count += 1
print('movie dictionary building done')

user_input = Input(batch_shape = (None, 1))
movie_input = Input(batch_shape = (None, 1))

model = model_mf(user_input, movie_input, attribute, user_count, movie_count)

model.compile(loss = 'mean_squared_error', optimizer = 'Adam', metrics = ['accuracy'])

print(model.summary())

print('start training the movie rating model')

y = np.asarray(train_data.iloc[:, 3])
    
x_user = np.asarray(train_data.iloc[:, 1]).reshape((train_data.shape[0], 1))
x_movie = np.asarray(train_data.iloc[:, 2]).reshape((train_data.shape[0], 1))

model.fit([x_user, x_movie], y, batch_size = 500, epochs = 100)

model.save('mf_model.h5')

print('all done')
