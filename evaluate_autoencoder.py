from io import open
from time import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras import regularizers
from keras import backend as K
from keras import optimizers

import os
import numpy as np
import re
import csv

GLOVE_DIR           = './embedding'
GLOVE_FILE          = 'glove.6B.50d.txt';
EMBEDDING_DIM       = 50
MAX_SENTENCE_LENGTH = 37

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, GLOVE_FILE))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print("Found %s word vectors." % len(embeddings_index))

conv1d_encoder = load_model('./models/autoencoder/cnn1d_encoder_2x50.h5')
conv1d_denoising_encoder = load_model('./models/autoencoder/cnn1d_encoder_2x50_denoise.h5')

testing_data = './data/testing_data.csv'
keys = ['a)', 'b)', 'c)', 'd)', 'e)']

with open(testing_data, 'r') as f:
    reader = csv.DictReader(f)
    
    sent_vec = np.zeros((1040*5, 925))
    for i, row in enumerate(reader):
        question = re.sub(r"\.", " . ", row['question'])
        question = re.sub(r"\s{2,}", " ", question)
        question = question.strip().lower().split()

        choices = [row[x] for x in keys]
        
        pos = 0
        for word_index, word in enumerate(question):
            if word == '_____':
                pos = word_index

        questions = [' '.join(question).replace('_____', word) for word in choices]

        for j, q in enumerate(questions):
            vec = []
            tokens = q.split()

            diff = MAX_SENTENCE_LENGTH - len(tokens)
            while diff > 0:
                vec.append(np.zeros(EMBEDDING_DIM))
                diff -= 1

            for token in tokens:
                token = clean_str(token)
                embedding_vector = embeddings_index.get(token)
                if embedding_vector is not None:
                    vec.append(embedding_vector)
                else:
                    vec.append(np.zeros(EMBEDDING_DIM))

            vec = np.array(vec)
            vec = np.reshape(vec, (1, MAX_SENTENCE_LENGTH*EMBEDDING_DIM, 1))
            print q
            print vec.shape
            sent_vec[i*5+j] = np.reshape(conv1d_denoising_encoder.predict(vec), (1, 925))
            print sent_vec[i*5+j].shape
    np.savetxt('./data/test_conv1d_denoise.txt', sent_vec, fmt='%.4f', delimiter=',')