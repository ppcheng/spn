from io import open

import os
import glob
import time

import numpy as np
import gensim

def ngram2vec(ngram, ngram_data_dir, output_matrix_dir, max_line_limit, embedding_file, is_embedding_binary):
    start_time = time.time()
    files = glob.glob(os.path.join(ngram_data_dir, "*.TXT"))
    print "Number of files to process:", len(files)

    print "Start importing the embedding file:", embedding_file, "in binary mode =", is_embedding_binary
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=is_embedding_binary)
    print "Finish importing the embedding file"
    dim = word_vectors['word'].size
    print "Size of a embedding vector is", dim

    if not os.path.exists(output_matrix_dir):
        os.makedirs(output_matrix_dir)

    for file_index, file_path in enumerate(files):
        print 'Progress:', file_index+1, "/", len(files)
        print 'Processing file:', file_path
        with open(file_path, "r", errors='ignore') as f:
            sent_vec = []

            for i, line in enumerate(f):
                if i >= max_line_limit:
                    break
                else:
                    tokens = line.strip().split()
                    for token in tokens:
                        if token in word_vectors:
                            sent_vec.append(word_vectors[token])
                        else:
                            sent_vec.append(np.random.uniform(-0.25, 0.25, dim))
            sent_vec = np.array(sent_vec)
            sent_vec = np.reshape(sent_vec, (-1, dim*ngram))
            np.savetxt(output_matrix_dir+'/'+os.path.basename(file_path), sent_vec, fmt='%.4f', delimiter=',')

    end_time = time.time()
    print "Time of execution:", end_time-start_time

google_word2vec_300d = './word_embedding/GoogleNews-vectors-negative300.bin'
glove_50d = './word_embedding/glove.6B.50d.word2vec.txt'
glove_100d = './word_embedding/glove.6B.100d.word2vec.txt'

ngram = 5
data = './data/Holmes_Training_Data_5gram'
output = './data/Holmes_Training_Data_5gram_100d'

max_line_limit = 250000
