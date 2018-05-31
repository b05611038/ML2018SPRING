import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
from keras.layers import Dense, GRU, Dropout, Input, Flatten
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import gensim
from gensim.models import Word2Vec

def rnn_model(model_input):
        x = GRU(512, return_sequences = True, recurrent_dropout = 0.5)(model_input)
        x = Dropout(0.5)(x)
        x = GRU(512, return_sequences = True, recurrent_dropout = 0.5)(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(1024, activation = 'hard_sigmoid')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation = 'hard_sigmoid')(x)
        x = Dropout(0.5)(x)
        x = Dense(2, activation = 'softmax')(x)
        
        model = Model(model_input, x)

        return model


train_label = pd.read_fwf(sys.argv[1], header = None)
train_nolabel = pd.read_fwf(sys.argv[2], header = None)
#test = pd.read_fwf('testing_data.txt')
#test.drop(test.index[0])

sentence = train_label.iloc[:, 2]
#sentence_test = test.iloc[:, 0]

label = np.asarray(train_label.iloc[:, 0], dtype = np.int)

print('start make the list of all word')
#input_sentence = [[] for i in range(len(sentence) + len(sentence_test))]
input_sentence = [[] for i in range(len(sentence))]
longest = 0
for i in range(len(sentence)):
    temp = sentence[i]
    temp = str(temp)
    temp = text_to_word_sequence(temp, filters = '"#$%&()*+,-./:;<=>@[\]^_`{|}~\t\n')

    longest_temp = len(temp)
    if longest_temp >= longest:
        longest = longest_temp
del temp
'''
for i in range(len(sentence_test)):
    temp = sentence_test[i].replace(str(i) + ',', '')
    temp = str(temp)
    temp = text_to_word_sequence(temp, filters = '"#$%&()*+,-./:;<=>@[\]^_`{|}~\t\n')

    longest_temp = len(temp)
    if longest_temp >= longest:
        longest = longest_temp
del temp
'''
longest = 39
print('the longest sentences have ' + str(longest) + ' words, the word2vec training data may be 2000000 * ' + str(longest))

for i in range(len(sentence)):
    temp = sentence[i]
    temp = text_to_word_sequence(temp, filters='"#$%&()*+,-./:;<=>@[\]^_`{|}~\t\n')

    fill_count = longest - len(temp)
    for j in range(fill_count):
        temp.append('')
    
    input_sentence[i] = temp
del temp
'''
for i in range(len(sentence_test)):
    temp = sentence_test[i].replace(str(i) + ',', '')
    temp = text_to_word_sequence(temp, filters='"#$%&()*+,-./:;<=>@[\]^_`{|}~\t\n')

    fill_count = longest - len(temp)
    for j in range(fill_count):
        temp.append('')

    input_sentence[len(sentence) + i] = temp
del temp
'''
print('making word list done\nstart training word2vec model')

w2v = Word2Vec(input_sentence, size = 200, window = 10, min_count = 5, workers = 4)

#w2v.train(input_sentence, total_examples = len(sentence) + len(sentence_test), epochs = 5)
w2v.train(input_sentence, total_examples = len(sentence), epochs = 5)

w2v.save('word2vec_model.bin')

print('word2vec model training done')

model_input = Input(batch_shape = (None, longest, 200))

print('start making the data matrix')
train_x = np.empty((len(sentence), longest, 200))
for j in range(len(sentence)):
    temp = np.empty((longest, 200))
    for k in range(longest):
        if input_sentence[j][k] in w2v.wv.vocab:
            temp[k] = np.asarray(w2v[str(input_sentence[j][k])])
        else:
            temp[k] = np.zeros(200)

    train_x[j] = temp

print('train_x ndarray creating done')

train_y = to_categorical(label)

model = rnn_model(model_input)

model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

print(model.summary())

print('start training RNN_model')
model.fit(x = train_x, y = train_y, batch_size = 500, epochs = 10)
   
model.save('rnn_model.h5')

print('all done')
