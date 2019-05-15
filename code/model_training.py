import pandas as pd
import numpy as np
import os
import sys
import json
from absl import flags
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# python model_training.py --vector_file onetwothree_jump_x.csv --label_file y_log.csv --config_file 300_50_001_adam_7_06_tanh_relu_05.json --model_name 123_300_50_001_adam_7_06_tanh_relu_05

flags.DEFINE_string('vector_file', None, 'File with training vectors.')
flags.DEFINE_string('label_file', None, 'File with label vector.')
flags.DEFINE_string('config_file', None, 'Config file with model parameters.')
flags.DEFINE_string('model_name', None, 'Named version of model')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

vector_file = FLAGS.vector_file
label_file = FLAGS.label_file
config_file = FLAGS.config_file
model_name = FLAGS.model_name

# model_dir=FLAGS.model_dir

print('reading data')
os.chdir('../data')
x_data = np.loadtxt(vector_file)
y_vector = np.loadtxt(label_file)
y_data = y_vector.reshape((-1,1))

row_ct = x_data.shape[0]
print(row_ct)

np.random.seed(42)
cut = int(0.85 * row_ct)
indices = np.random.permutation(row_ct)

print('splitting into train and test')
train_idx, test_idx = indices[:cut], indices[cut:]
x_train, x_test = x_data[train_idx, :], x_data[test_idx,:]
y_train, y_test = y_data[train_idx, :], y_data[test_idx,:]

input_shape = x_train.shape[1]

print('opening config file')
os.chdir('../configs')
with open(config_file, 'r') as json_file:
    params = json.load(json_file)

batch_sz = params['batch_size']
epoch_ct = params['epochs']
lr = params['learning_rate']
optimizer = params['optimizer']
if optimizer == 'adam':
    opt = Adam(lr=lr)
elif optimizer == 'rmsprop':
    opt = RMSprop(lr=lr)
elif optimizer == 'sgd':
    opt = SGD(lr=lr)
layer_ct = params['hidden_layers']
do = params['dropout']
h_act = params['hidden_act']
o_act = params['output_act']
scale = params['scaling']

print('building model')
model = Sequential()
for l in range(layer_ct):
    if l == 0:
        unit_ct = int(input_shape * scale)
        model.add(Dense(unit_ct, activation=h_act, input_dim=input_shape))
        if do > 0:
            model.add(Dropout(do))
            unit_ct * (1 - do)
    else:
        unit_ct = int(unit_ct * scale)
        model.add(Dense(unit_ct, activation=h_act))
model.add(Dense(1, activation=o_act))
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'acc'])

model.summary()

print('running model')
# train model and save history object
history = model.fit(x_train, y_train,
          batch_size=batch_sz,
          epochs=epoch_ct,
          validation_data=(x_test, y_test))

os.chdir('../models')

# save history of model as json file
with open(model_name + '_history.json', 'w') as hist_file:
    json.dump(history.history, hist_file)

# serialize model to JSON and save model format
model_json = model.to_json()
with open(model_name + '_model.json', 'w') as model_file:
    json.dump(model_json, model_file)

# serialize weights to HDF5 and save
model.save_weights(model_name + '_weights.h5')


import winsound
winsound.Beep(294, 500)
winsound.Beep(330, 500)
winsound.Beep(262, 500)
winsound.Beep(131, 500)
winsound.Beep(196, 500)