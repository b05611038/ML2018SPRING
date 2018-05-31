import sys
import csv
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import np_utils
from keras.models import Model, load_model
import gensim
from gensim.models import Word2Vec

mode = sys.argv[3]

test = pd.read_fwf(sys.argv[1])

print('start to load word2vec model')
w2v = Word2Vec.load('word2vec_model.bin')
print('word2vec model loading done')

sentence_test = test.iloc[:, 0]

print('start to load rnn_model')
model = load_model('rnn_model.h5')
print('rnn_model loading done')

print('start the process of prediction')
longest = 39
pre_array = np.empty((len(sentence_test), longest, 200))
for i in range(len(sentence_test)):
    temp = sentence_test[i]
    temp = text_to_word_sequence(temp, filters = '"#$%&()*+,-./:;<=>@[\]^_`{|}~\t\n')

    fill_count = longest - len(temp)
    for j in range(fill_count):
        temp.append('')

    temp_array = np.empty((longest, 200))
    for j in range(longest):
        if temp[j] in w2v.wv.vocab:
            temp_array[j] = w2v[temp[j]]
        else:
            temp_array[j] = np.zeros(200)
    pre_array[i] = temp_array 

prediction = model.predict(x = pre_array, batch_size = 300, verbose = 1)
prediction = np.argmax(prediction, axis = 1)

print('start outfile')
outfile = open(sys.argv[2], 'w')
lines = ['id,label\n']

for i in range(len(prediction)):
    lines.append(str(i) + ',' + str(prediction[i]) + '\n')

outfile.writelines(lines)
outfile.close()
print('all done')
