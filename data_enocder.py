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

SEED_NO             = 20321984
MAX_NUM_WORDS       = 20000
MAX_SENTENCE_LENGTH = 37
TRAIN_DATA          = './data/train.txt'
VALIDATION_PORTION  = 0.05
GLOVE_DIR           = './embedding'
GLOVE_FILE          = 'glove.6B.50d.txt';
EMBEDDING_DIM       = 50

np.random.seed(SEED_NO)

texts = []

with open(TRAIN_DATA, 'r') as f:
    lines = f.read()

    lines = lines.split('\n')
    for index, line in enumerate(lines):
        if (line != ''):
            texts.append(line.encode('ascii'))

print "Number of texts: ", len(texts)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SENTENCE_LENGTH)
print("Shape of data tensor:", data.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]

nb_validation_samples = int(VALIDATION_PORTION * data.shape[0])

x_train = data[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]

print "Training Data Size:", len(x_train)
print "Validation Data Size:", len(x_val)

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, GLOVE_FILE))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print("Found %s word vectors." % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

x_train = np.stack([np.reshape(np.stack([embedding_matrix[word] for word in sentence]), MAX_SENTENCE_LENGTH*EMBEDDING_DIM) for sentence in x_train])
x_val = np.stack([np.reshape(np.stack([embedding_matrix[word] for word in sentence]), MAX_SENTENCE_LENGTH*EMBEDDING_DIM) for sentence in x_val])
print "Training Dataset Size:", x_train.shape
print "Validation Dataset Size:", x_val.shape

conv1d_encoder = load_model('./models/autoencoder/cnn1d_encoder_2x50.h5')
conv1d_denoising_encoder = load_model('./models/autoencoder/cnn1d_encoder_2x50_denoise.h5')

x_train_conv1d = np.stack([conv1d_encoder.predict(np.reshape(vec, (1, MAX_SENTENCE_LENGTH*EMBEDDING_DIM, 1))) for vec in x_train])
x_val_conv1d = np.stack([conv1d_encoder.predict(np.reshape(vec, (1, MAX_SENTENCE_LENGTH*EMBEDDING_DIM, 1))) for vec in x_val])

x_train_conv1d = np.reshape(x_train_conv1d, (len(x_train_conv1d), 925))
x_val_conv1d = np.reshape(x_val_conv1d, (len(x_val_conv1d), 925))
np.savetxt('./data/train_conv1d.txt', x_train_conv1d, fmt='%.4f', delimiter=',')
np.savetxt('./data/val_conv1d.txt', x_train_conv1d, fmt='%.4f', delimiter=',')
print "Training (Conv1D) Dataset Size:", x_train_conv1d.shape
print "Validation (Conv1D) Dataset Size:", x_val_conv1d.shape


x_train_conv1d_denoise = np.stack([conv1d_denoising_encoder.predict(np.reshape(vec, (1, MAX_SENTENCE_LENGTH*EMBEDDING_DIM, 1))) for vec in x_train])
x_val_conv1d_denoise = np.stack([conv1d_denoising_encoder.predict(np.reshape(vec, (1, MAX_SENTENCE_LENGTH*EMBEDDING_DIM, 1))) for vec in x_val])

x_train_conv1d_denoise = np.reshape(x_train_conv1d_denoise, (len(x_train_conv1d_denoise), 925))
x_val_conv1d_denoise = np.reshape(x_val_conv1d_denoise, (len(x_val_conv1d_denoise), 925))
np.savetxt('./data/train_conv1d_denoise.txt', x_train_conv1d, fmt='%.4f', delimiter=',')
np.savetxt('./data/val_conv1d_denoise.txt', x_train_conv1d, fmt='%.4f', delimiter=',')
print "Training (Conv1D-Denoise) Dataset Size:", x_train_conv1d_denoise.shape
print "Validation (Conv1D-Denoise) Dataset Size:", x_val_conv1d_denoise.shape