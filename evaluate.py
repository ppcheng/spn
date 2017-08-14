from io import open

import csv
import re

import numpy as np
import gensim

from tachyon.SPN2 import SPN

test_answer_file = './data/test_answer.csv'
answer_arr = []

with open(test_answer_file, 'r') as data:
    reader = csv.DictReader(data)

    for i, row in enumerate(reader):
        answer_arr.append(row['answer']) 


print len(answer_arr)


spn = SPN()
spn.make_fast_model_from_file('./models/holmes.spn.set1nset2_epoch1.txt', cont=True, random_weights=False)
spn.start_session()

test_data = './data/test_matrix_5gram_300d'
spn.add_data(test_data, 'test', cont=True)
test = spn.data.test
spn.evaluate(test, solution_set=answer_arr, minibatch_size=1)
