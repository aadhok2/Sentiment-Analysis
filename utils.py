import pickle as pkl
import csv
import tensorflow as tf
import numpy as np
import random


# Function to split a document into a list of tokens
# Arguments:
# text: A string containing input document
# Returns: tokens (list)
# Where, tokens (list) is a list of tokens that the document is split into
# Text is already tokenized using nltk.tokenize.word_tokenize and tokens are
# space separated so it should be enough to just split on space.
def get_tokens(text):
	return text.split(' ')


# Function to load movie reviews dataset
# Arguments: path of csv file containing data
# Returns: a list of dictionaries, such as, each dictionary contains the following fields:
# id (a unique identifier), label (pos/neg class label), split (specifying whether the review is in train/val/test split), text (the review)
def load_data(filepath):
    data = []
    with open(filepath, encoding='utf-8') as fin:
        reader = csv.reader(fin)
        header = next(reader)
        for row in reader:
            record = {}
            for idx, field in enumerate(header):
                record[field] = row[idx]
                if field == 'id':
                    record[field] = int(record[field])
            data.append(record)
    return data


# Function to load word vectors pre-trained on Google News
# Arguments: None
# Returns: w2v (dict)
# Where, w2v (dict) is a dictionary with words as keys and vectors as values
def load_w2v():
    with open('w2v.pkl', 'rb') as fin:
        return pkl.load(fin)


# Function to reset seeds for the sake of consistency
def reset_seeds():
    SEED = 100
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)
