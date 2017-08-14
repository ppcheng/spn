import os
import re
import csv
import glob
import pickle
import string
import collections

import numpy as np
from io import open

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_test_data():
    avg_len = 0
    max_len = 0
    max_sequence = []
    length_dist = {}
    dist = {
        '10': 0,
        '20': 0,
        '30': 0,
        '40': 0 
    }
    data = './data/testing_data.csv'
    with open(data, 'r') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            question = re.sub(r"\.", " . ", row['question'])
            question = re.sub(r"\s{2,}", " ", question)
            question = question.strip().lower().split()

            length = len(question)
            avg_len += length

            if length in length_dist:
                length_dist[length] += 1
            else:
                length_dist[length] = 1

            if length < 10:
                dist['10'] += 1
            elif length >= 10 and length < 20:
                dist['20'] += 1
            elif length >= 20 and length < 30:
                dist['30'] += 1
            else:
                dist['40'] += 1

            if length > max_len:
                max_len = length
                max_sequence = question
        print "Number of test instance:", i+1
        print "Total Sentence Length:", avg_len
        print "Average Sentence Length:", avg_len/(i+1)
        print "Maximum Sentence Length:", max_len
        print "Sentence of Maixmum Length:", max_sequence
        print length_dist
        print dist
load_test_data()

def get_train_data():
    data_dir = "./data/Holmes_Training_Data/"
    #output_dir = './data/Holmes_Training_Data_Clean/'

    input_files = glob.glob(os.path.join(data_dir, "*.TXT"))
    print "Total number of files: %s" % len(input_files)

    num_training_sample = 50000
    writer = open('./data/train.txt', 'w')

    for input_file in input_files:
        if num_training_sample > 0:
            with open(input_file, "r", errors="ignore") as f:
                print "Reading from: ", input_file
                
                data = f.read()
                
                # text cleaning or make them lower case, etc.
                data = clean_str(data) 
                
                sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?;!])\s",data)

                for index, sentence in enumerate(sentences):
                    s = re.sub(r"\.", " . ", sentence)
                    s = re.sub(r"\s{2,}", " ", s)
                    tokens = s.strip().split()

                    if len(tokens) >= 6 and len(tokens) <= 37 and num_training_sample > 0:
                        writer.write(u"%s\n" % (s))
                        num_training_sample -= 1
        else:
            break;
get_train_data()