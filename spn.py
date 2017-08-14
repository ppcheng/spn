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

from tachyon.SPN2 import SPN

training_matrix_dir = './data/Holmes_Training_Data_5gram_50d_matrix'

train_set = glob.glob(os.path.join(training_matrix_dir, "*.TXT"))

# make an SPN holder
spn = SPN()

# include training and testing data
# spn.add_data('./data/Holmes_Training_Data_5gram_matrix/1ADAM10.TXT', 'train', cont=True)
# spn.add_data('./data/Holmes_Training_Data_5gram_matrix/ZENDA10.TXT', 'valid', cont=True)

# create a valid sum product network
sum_branch_factor = (2, 4)
prod_branch_factor = (5, 40)
variables = 250

#continuous model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=True, classify=False, tensorboard_dir="./logs")

# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = False
gd = True
count = False #take a step of counting then a step of gradient descent

#large minibatches with a small number at a time
minibatch_size=100

# other stuff
epochs = 5

# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test

for e in epochs:
	for f in train_set:
		spn.add_data(f, 'train', cont=True)
		train = spn.data.train
		spn.train(epochs, train, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size)
	

# test_loss = spn.evaluate(test, minibatch_size=1)
# print 'Loss:', test_loss
# Loss: 4.513

spn.model.unbuild_fast_variables() #pull weights from tensorflow.
spn.model.save("./models/holmes.spn.50d_epoch_5.txt")