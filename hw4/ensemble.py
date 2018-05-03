import sys
import os
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Average, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input

def cnn16(model_input):
	s = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(model_input)
	s = Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same')(s)
	s = MaxPooling2D((2, 2))(s)
	s = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(s)
	s = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(s)
	s = Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same')(s)
	s = MaxPooling2D((2, 2))(s)
	s = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(s)
	s = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(s)
	s = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', padding = 'same')(s)
	s = MaxPooling2D((2, 2))(s)
	s = Flatten()(s)
	s = Dense(1024, activation = 'relu')(s)
	s = Dropout(0.5)(s)
	s = Dense(1024, activation = 'relu')(s)
	s = Dropout(0.5)(s)
	s = Dense(7, activation = 'softmax')(s)

	model = Model(model_input, s)

	return model


def ensembleModels(models, model_input):
	outputs = [model.outputs[0] for model in models]
	output = Average()(outputs)

	modelEns = Model(model_input, output)

	return modelEns

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
set_session(tf.Session( config = config))

model_input = Input(shape = (48, 48, 1))

cnn_model1 = cnn16(model_input)
cnn_model2 = cnn16(model_input)
cnn_model3 = cnn16(model_input)
cnn_model4 = cnn16(model_input)

print(cnn_model1.summary())
cnn_model1.load_weights('model_ver1_weight.hd5f')
cnn_model2.load_weights('model_ver2_weight.hd5f')
cnn_model3.load_weights('model_ver3_weight.hd5f')
cnn_model4.load_weights('model_ver4_weight.hd5f')

models = [cnn_model1, cnn_model2, cnn_model3, cnn_model4]

ensModel = ensembleModels(models, model_input)

ensModel.save('ensModel.h5')
