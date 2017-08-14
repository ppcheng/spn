from io import open
from time import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import regularizers
from keras import backend as K
from keras import optimizers

import os
import numpy as np

def convolution_autoencoder():
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
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SENTENCE_LENGTH)
    print('Shape of data tensor:', data.shape)

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

    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SENTENCE_LENGTH,
                                trainable=False)
    x_train = np.stack([np.reshape(np.stack([embedding_matrix[word] for word in sentence]), MAX_SENTENCE_LENGTH*EMBEDDING_DIM) for sentence in x_train])

    x_val = np.stack([np.reshape(np.stack([embedding_matrix[word] for word in sentence]), MAX_SENTENCE_LENGTH*EMBEDDING_DIM) for sentence in x_val])

    x_train = np.reshape(x_train, (len(x_train), MAX_SENTENCE_LENGTH*EMBEDDING_DIM, 1))
    x_val = np.reshape(x_val, (len(x_val), MAX_SENTENCE_LENGTH*EMBEDDING_DIM, 1))
    print "Training Dataset Size:", x_train.shape
    print "Validation Dataset Size:", x_val.shape

    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_val_noisy = x_val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_val.shape)

    sequence_input = Input(shape=(MAX_SENTENCE_LENGTH*EMBEDDING_DIM, 1), dtype='float32')
    h = Conv1D(1, 2*EMBEDDING_DIM, activation='relu', padding='same')(sequence_input)
    # h = MaxPooling1D(2, padding='same')(h)
    # h = Conv1D(2, 2*EMBEDDING_DIM, activation='relu', padding='same')(h)
    encoded = MaxPooling1D(2, padding='same')(h)
    # h = Conv1D(2, 2*EMBEDDING_DIM, activation='relu', padding='same')(encoded)
    # h = UpSampling1D(5)(h)
    h = Conv1D(1, 2*EMBEDDING_DIM, activation='relu', padding='same')(encoded)
    h = UpSampling1D(2)(h)
    decoded = Conv1D(1, 2*EMBEDDING_DIM, activation='relu', padding='same')(h)

    # this model maps an input to its reconstruction
    autoencoder = Model(sequence_input, decoded)

    TIME = time()

    autoencoder.compile(optimizer='adam', loss='mse')
    tensorboard = TensorBoard(log_dir="logs/{0}_{1}.log".format('conv1d_autoencoder', TIME), 
                    histogram_freq=0, write_graph=True, write_images=True)

    autoencoder.fit(x_train_noisy, x_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_val_noisy, x_val), verbose=1, callbacks=[tensorboard])

    encoder = Model(sequence_input, encoded)
    
    # Save trained autoencoder model
    
    autoencoder.save('./autoencoder/cnn1d_{0}.h5'.format(TIME), overwrite=True)

    # Save encoder with weights
    encoder.save('./autoencoder/cnn1d_encoder_{0}.h5'.format(TIME), overwrite=True)

    encoded_imgs = encoder.predict(x_val)
    decoded_imgs = autoencoder.predict(x_val)

    for i in range(5):
        print encoded_imgs[i].shape
        print decoded_imgs[i].shape

convolution_autoencoder()