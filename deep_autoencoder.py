from io import open
from time import time

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model
from keras.callbacks import TensorBoard
from keras import regularizers

import os
import numpy as np

texts = []
np.random.seed(20321984)

with open('./data/train.txt', 'r') as f:
    lines = f.read()

    lines = lines.split('\n')
    for index, line in enumerate(lines):
        if (line != ''):
            texts.append(line.encode('ascii'))

print "Number of texts: ", len(texts)

# texts = [s.encode('ascii') for s in texts]

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=37)

print('Shape of data tensor:', data.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]

nb_validation_samples = int(0.05 * data.shape[0])

x_train = data[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]

print "Training Data Size:", len(x_train)
print "Validation Data Size:", len(x_val)

GLOVE_DIR = './embedding'
embeddings_index = {}

f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 50

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=37,
                            trainable=False)


x_train = np.stack([np.reshape(np.stack([embedding_matrix[word] for word in sentence]), 37*50) for sentence in x_train])

x_val = np.stack([np.reshape(np.stack([embedding_matrix[word] for word in sentence]), 37*50) for sentence in x_val])

sequence_input = Input(shape=(37*50,), dtype='float32')
h1 = Dense(900, activation='relu')(sequence_input)
h2 = Dense(450, activation='relu')(h1)
# encoded = Dense(300, activation='relu', activity_regularizer=regularizers.l1(1e-5))(h2)
encoded = Dense(300, activation='relu', activity_regularizer=regularizers.l1(1e-5))(h2)
h2 = Dense(450, activation='relu')(encoded)
h1 = Dense(900, activation='relu')(h2)
decoded = Dense(37*50, activation='sigmoid')(h1)

# this model maps an input to its reconstruction
autoencoder = Model(sequence_input, decoded)
encoder = Model(sequence_input, encoded)
encoded_input = Input(shape=(300,))
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

autoencoder.compile(optimizer='adam', loss='mse')
tensorboard = TensorBoard(log_dir="logs/deep_autoencoder_{0}".format(time()), histogram_freq=0,
                          write_graph=True, write_images=True)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_val, x_val), verbose=1, callbacks=[tensorboard])

TIME = time()

# Save trained autoencoder model
autoencoder.save('./autoencoder/deep_{0}.h5'.format(TIME), overwrite=True)

# Save encoder with weights
encoder.save('./autoencoder/deep_encoder_{0}.h5'.format(TIME), overwrite=True)

# Save decoder
decoder.save('./autoencoder/deep_decoder_{0}.h5'.format(TIME), overwrite=True)

encoded_imgs = encoder.predict(x_val)
decoded_imgs = decoder.predict(encoded_imgs)

for i in range(5):
    print encoded_imgs[i].shape
    print decoded_imgs[i].shape