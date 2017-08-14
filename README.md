## Project Structure
1. `models/autoencoder`: contain all the trained convolution autoencoder models
2. `models/spn`: 

## Data
Link to the data set: https://drive.google.com/drive/folders/0B3au6xxMXb02aHRiSTk2R3RUM1E?usp=sharing

## tachyon
- Can work on tensorflow v1+

## Pre-trained Word Embedding
1. GloVe: https://nlp.stanford.edu/projects/glove/
2. Facebook: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
3. https://github.com/3Top/word2vec-api
4. https://github.com/Kyubyong/wordvectors
5. http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/

How to convert glove embedding to word2vec format that gensim can read

```
python -m gensim.scripts.glove2word2vec –i <GloVe vector file> –o <Word2vec vector file>
```

## CNN For NLP
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow

http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp

## Reference
https://github.com/zihanchen/Conversational-Agent/blob/master/spn_experiment.py
https://github.com/chiawen/sentence-completion/blob/master/train.py
https://github.com/chiawen/sentence-completion/blob/master/utils.py
https://github.com/KalraA/Tachyon


https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
https://blog.keras.io/building-autoencoders-in-keras.html
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py
https://chunml.github.io/ChunML.github.io/project/Sequence-To-Sequence/

