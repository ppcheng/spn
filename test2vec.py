from io import open

import csv
import re

import numpy as np
import gensim

def clean_str(string):
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    """
    string = re.sub(r"\[([^\]]+)\]", " ", string)
    string = re.sub(r"\(([^\)]+)\)", " ", string)
    string = re.sub(r"[^A-Za-z0-9,!?.;]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r";", " ; ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

testing_data = './data/testing_data.csv'
keys = ['a)', 'b)', 'c)', 'd)', 'e)']
embedding_file = './word_embedding/GoogleNews-vectors-negative300.bin'

glove_50d = './word_embedding/glove.6B.50d.word2vec.txt'
glove_100d = './word_embedding/glove.6B.100d.word2vec.txt'

print "Import word embedding file"
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(glove_50d)
dim = word_vectors['word'].size

with open(testing_data, 'r') as data:
    reader = csv.DictReader(data)

    sent_vec = []
    for i, row in enumerate(reader):
        test_instance = {'id':row['id'], 'candidates':[] }
        question = re.sub(r"\.", " . ", row['question'])
        question = re.sub(r"\s{2,}", " ", question)
        question = question.strip().lower().split()

        pos = 0
        for word_index, word in enumerate(question):
            if word == '_____':
                pos = word_index

        print question[pos-2:pos+3]
            # print word_index, word
        choices = [row[x] for x in keys]
        questions = [' '.join(question[pos-2:pos+3]).replace('_____', word) for word in choices]

        for j in questions:
            tokens = j.split()
            for token in tokens:
                token = clean_str(token)
                if token in word_vectors:
                    sent_vec.append(word_vectors[token])
                else:
                    sent_vec.append(np.random.uniform(-0.25, 0.25, dim))
        
    sent_vec = np.array(sent_vec)
    sent_vec = np.reshape(sent_vec, (-1, 250))
    np.savetxt('./data/test_matrix_5gram_50d', sent_vec, fmt='%.4f', delimiter=',')