import pandas as pd
import numpy as np
import os
import json
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


os.chdir('../data')
train_df = pd.read_csv('train_pairs.csv')

# build initial dictionary and save as json
def dict_build(data, json_file):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    pair_index = tokenizer.word_index

    with open(json_file, 'w') as dict_file:
        json.dump(pair_index, dict_file)

# three_lists = list(train_df['one_jump_pairs']) + list(train_df['two_jump_pairs']) + list(train_df['three_jump_pairs'])
# dict_build(three_lists, 'pair_dict_full.json')

with open('pair_dict_full.json', 'r') as dict_file:
    pair_index_full = json.load(dict_file)

pair_ct = len(pair_index_full)

def vectorize_pairs(pairs, pair_ct):
    results = np.zeros((len(pairs), pair_ct))
    for i, rw in enumerate(pairs):
        for pair in rw.split(' '):
            results[i, pair_index_full[pair.lower()] - 1] = 1
    return results

one_jump_vectors = vectorize_pairs(train_df['one_jump_pairs'], pair_ct)
two_jump_vectors = vectorize_pairs(train_df['two_jump_pairs'], pair_ct)
three_jump_vectors = vectorize_pairs(train_df['three_jump_pairs'], pair_ct)

bin_list = train_df['binary']
bin_len = len(bin_list[0])
binary = np.zeros((len(bin_list), bin_len))
for i, rw in enumerate(bin_list):
    for j, digit in enumerate(rw):
        binary[i, j] = int(digit)

one = np.concatenate((binary, one_jump_vectors), axis=1)
onetwo = np.concatenate((one, two_jump_vectors), axis=1)
onetwothree = np.concatenate((onetwo, three_jump_vectors), axis=1)

print(binary.shape)
print(one.shape)
print(onetwo.shape)
print(onetwothree.shape)

np.savetxt('one_jump_x.csv', one)
np.savetxt('onetwo_jump_x.csv', onetwo)
np.savetxt('onetwothree_jump_x.csv', onetwothree)

labels = np.asarray(train_df['kd'])
labels_log = -np.log10(labels/1000000000)

print(labels.shape)
print(labels_log.shape)

np.savetxt('y.csv', labels)
np.savetxt('y_log.csv', labels_log)

import winsound
winsound.Beep(400, 350)
winsound.Beep(400, 200)
winsound.Beep(400, 200)
winsound.Beep(500, 600)