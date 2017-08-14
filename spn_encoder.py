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
from time import time

from tachyon.SPN2 import SPN
current = time()
print "Start at:", current

spn = SPN()

sum_branch_factor = (2, 4)
prod_branch_factor = (20, 40)
variables = 925

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
epochs = 50

spn.add_data('./data/train_conv1d_denoise.txt', 'train', cont=True)
spn.add_data('./data/val_conv1d_denoise.txt', 'valid', cont=True)
train = spn.data.train
valid = spn.data.valid

spn.train(epochs, train, valid_data=spn.data.valid, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size)

spn.model.unbuild_fast_variables() #pull weights from tensorflow.
spn.model.save("./models/holmes.spn.conv1d_denoise.txt")

print "Total Training Time:", time()-current