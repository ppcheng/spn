import os
import re
import csv
import glob
import pickle
import string
import collections

import numpy as np
import gensim

from io import open

data_dir = './data/Holmes_Training_Data_5gram'

embedding_file = './word_embedding/glove.6B.50d.word2vec.txt'

training_matrix_dir = './data/Holmes_Training_Data_5gram_50d_matrix'

input_files = glob.glob(os.path.join(data_dir, "*.TXT"))

line_limit = 5000

#word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_file)

dim = word_vectors['word'].size

def convert(input_files):
    if not os.path.exists(training_matrix_dir):
        os.makedirs(training_matrix_dir)

    for input_file in input_files:
        print('Processing file: ' + input_file)
        with open(input_file, "r", errors='ignore') as f:
            sentences = f.readlines()
            sent_vec = []

            for index, sent in enumerate(sentences):
                print index
                tokens = sent.strip().split()
                
                for token in tokens:
                    if token in word_vectors:
                        sent_vec.append(word_vectors[token])
                    else:
                        sent_vec.append(np.random.uniform(-0.25, 0.25, dim))
                
            sent_vec = np.array(sent_vec)
            sent_vec = np.reshape(sent_vec, (-1, 250))
            np.savetxt(training_matrix_dir+'/'+os.path.basename(input_file), sent_vec, fmt='%.4f', delimiter=',')
                
convert(input_files)