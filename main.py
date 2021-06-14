import tensorflow as tf
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
import itertools as it
import numpy as np
import random
from utils import *


def tokenize_docs(docs):
    #stopwords = get_stopwords()
    print("Hi")
    tokens = [[t for t in get_tokens(d)] for d in docs]
    return tokens
# Function to get word2vec representations
#
# Arguments:
# reviews: A list of strings, each string represents a review
#
# Returns: mat (numpy.ndarray) of size (len(reviews), dim)
# mat is a two-dimensional numpy array containing vector representation for ith review (in input list reviews) in ith row
# dim represents the dimensions of word vectors, here dim = 300 for Google News pre-trained vectors
def w2v_rep(reviews):
    dim = 300
    mat = np.zeros((len(reviews), dim))
    tokens = tokenize_docs(reviews)
    w2v = load_w2v()
    for idx, doc_tokens in enumerate(tokens):
        valid_tokens = 0
        for t in doc_tokens:
            if t in w2v:
                mat[idx] += w2v[t]
                valid_tokens += 1
        if valid_tokens > 0:
            mat[idx] = mat[idx] / valid_tokens
    #print('size',mat.shape)
    return mat



# Function to build a feed-forward neural network using tf.keras.Sequential model. You should build the sequential model
# by stacking up dense layers such that each hidden layer has 'relu' activation. Add an output dense layer in the end
# containing 1 unit, with 'sigmoid' activation, this is to ensure that we get label probability as output
#
# Arguments:
# params (dict): A dictionary containing the following parameter data:
#					layers (int): Number of dense layers in the neural network
#					units (int): Number of units in each dense layer
#					loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#					optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#
# Returns:
# model (tf.keras.Sequential), a compiled model created using the specified parameters
def build_nn(params):
    model = tf.keras.Sequential()
    # [YOUR CODE HERE]
    #print(params)
    for x in range(params['layers']):
        model.add(layers.Dense(params['units'], activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))
    model.compile
    return model


# Function to select the best parameter combination based on accuracy by evaluating all parameter combinations
# This function should train on the training set (X_train, y_train) and evluate using the validation set (X_val, y_val)
#
# Arguments:
# params (dict): A dictionary containing parameter combinations to try:
#					layers (list of int): Each element specifies number of dense layers in the neural network
#					units (list of int): Each element specifies the number of units in each dense layer
#					loss (list of string): Each element specifies the type of loss to optimize ('binary_crossentropy' or 'mse)
#					optimizer (list of string): Each element specifies the type of optimizer to use while training ('sgd' or 'adam')
#					epochs (list of int): Each element specifies the number of iterations over the training set
# X_train (numpy.ndarray): A matrix containing w2v representations for training set of shape (len(reviews), dim)
# y_train (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_train of shape (X_train.shape[0], )
# X_val (numpy.ndarray): A matrix containing w2v representations for validation set of shape (len(reviews), dim)
# y_val (numpy.ndarray): A numpy vector containing (0/1) labels corresponding to the representations in X_val of shape (X_val.shape[0], )
#
# Returns:
# best_params (dict): A dictionary containing the best parameter combination:
#	    				layers (int): Number of dense layers in the neural network
#	 	     			units (int): Number of units in each dense layer
#	 					loss (string): The type of loss to optimize ('binary_crossentropy' or 'mse)
#						optimizer (string): The type of optimizer to use while training ('sgd' or 'adam')
#						epochs (int): Number of iterations over the training set
def find_best_params(params, X_train, y_train, X_val, y_val):
    best_params = dict()
    #print(params)
    # Note that you don't necessarily have to use this loop structure for your experiments
    # However, you must call reset_seeds() right before you call build_nn for every parameter combination
    # Also, make sure to call reset_seeds right before every model.fit call

    # Get all parameter combinations (a list of dicts)
    # [YOUR CODE HERE]
    keys, values = zip(*params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in it.product(*values)]
    param_combinations = permutations_dicts
    max = 0
    #print(permutations_dicts)

    # Iterate over all combinations using one or more loops
    for param_combination in param_combinations:
        # Reset seeds and build your model
        #print(param_combination)
        reset_seeds()
        model = build_nn(param_combination)
        model.compile(optimizer=param_combination['optimizer'],loss=param_combination['loss'])
        reset_seeds()
        model.fit(X_train,y_train,epochs=param_combination['epochs'])
        #predict probabilities for test set
        yhat_probs = model.predict(X_val, verbose=0)
        # predict crisp classes for test set
        yhat_classes = model.predict_classes(X_val, verbose=0)
        yhat_probs = yhat_probs[:, 0]
        yhat_classes = yhat_classes[:, 0]
        # accuracy: (tp + tn) / (p + n)
        acc = accuracy_score(y_val, yhat_classes)
        #print('Accuracy: %f' % acc)
        if(acc > max):
            max = acc
            best_params = param_combination
    reset_seeds()
    model2 = build_nn(best_params)
    model2.compile(optimizer=best_params['optimizer'],loss=best_params['loss'])
    reset_seeds()
    model2.fit(X_train,y_train,epochs=best_params['epochs'])
    # predict probabilities for test set
    yhat_probs = model2.predict(X_val, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model2.predict_classes(X_val, verbose=0)
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_val, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_val, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_val, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_val, yhat_classes)
    print('F1 score: %f' % f1)
    return best_params


# Function to convert probabilities into pos/neg labels
#
# Arguments:
# probs (numpy.ndarray): A numpy vector containing probability of being positive
#
# Returns:
# pred (numpy.ndarray): A numpy vector containing pos/neg labels such that ith value in probs is mapped to ith value in pred
# 						A value is mapped to pos label if it is >=0.5, neg otherwise
def translate_probs(probs):
	# [YOUR CODE HERE]
    print(probs)
    pred = np.repeat('pos', probs.shape[0])
    for i in range(len(probs)):
        if probs[i] >= 0.5:
            pred[i] = "pos"
        else:
            pred[i] = "neg"
    print(pred)
    return pred


# Use the main function to test your code when running it from a terminal
# Sample code is provided to assist with the assignment, it is recommended
# that you do not change the code in main function for this assignment
# You can run the code from termianl as: python3 q3.py
# It should produce the following output and 2 files (q1-train-rep.npy, q1-pred.npy):
#
# $ python3 q1.py
# Best parameters: {'layers': 1, 'units': 8, 'loss': 'binary_crossentropy', 'optimizer': 'adam', 'epochs': 1}

def main():
    # Load dataset
    data = load_data('movie_reviews.csv')

    # Extract list of reviews from the training set
    # Note that since data is already sorted by review IDs, you do not need to sort it again for a subset
    train_data = list(filter(lambda x: x['split'] == 'train', data))
    reviews_train = [r['text'] for r in train_data]

    # Compute the word2vec representation for training set
    X_train = w2v_rep(reviews_train)
    # Save these representations in q1-train-rep.npy for submission
    np.save('q1-train-rep.npy', X_train)

    # Write your code here to extract representations for validation (X_val) and test (X_test) set
    # Also extract labels for training (y_train) and validation (y_val)
    # Use 1 to represent 'pos' label and 0 to represent 'neg' label
    # [YOUR CODE HERE]
    val_data = list(filter(lambda x: x['split'] == 'val', data))
    reviews_val = [r['text'] for r in val_data]
    X_val = w2v_rep(reviews_val)
    
    
    test_data = list(filter(lambda x: x['split'] == 'test', data))
    reviews_test = [r['text'] for r in test_data]
    X_test = w2v_rep(reviews_test)
    
    
    train2_data = list(filter(lambda x: x['split'] == 'train', data))
    labels_train = [r['label'] for r in train2_data]
    y_train = labels_train
    for i in range(0, len(y_train)): 
        y_train[i] = int(y_train[i]) 
    y_train = np.array(y_train)
    
    
    val2_data = list(filter(lambda x: x['split'] == 'val', data))
    labels_val = [r['label'] for r in val2_data]
    y_val = labels_val
    for i in range(0, len(y_val)): 
        y_val[i] = int(y_val[i]) 
    y_val = np.array(y_val)
    
    
    #print(y_train)

    # Build a feed forward neural network model with build_nn function
    params = {
        'layers': 1,
        'units': 8,
        'loss': 'binary_crossentropy',
        'optimizer': 'adam'
    }
    reset_seeds()
    model = build_nn(params)

    # Function to choose best parameters
    # You should use build_nn function in find_best_params function
    params = {
        'layers': [1, 3],
        'units': [8, 16, 32],
        'loss': ['binary_crossentropy', 'mse'],
        'optimizer': ['sgd', 'adam'],
        'epochs': [1, 5, 10]
    }
    best_params = find_best_params(params, X_train, y_train, X_val, y_val)

    # Save the best parameters in q1-params.csv for submission
    print("Best parameters: {0}".format(best_params))

    # Build a model with best parameters and fit on the training set
    # reset_seeds function must be called immediately before build_nn and model.fit function
    # Uncomment the following 4 lines to call the necessary functions
    reset_seeds()
    model = build_nn(best_params)
    reset_seeds()
    model.fit(X_train, y_train, epochs=best_params['epochs'])

    # Use the model to predict labels for the validation set (uncomment the line below)
    pred = model.predict(X_val).flatten()


    # Write code here to evaluate model performance on the validation set
    # You should compute precision, recall, f1, accuracy
    # Save these results in q1-res.csv for submission
    # Can you use translate_probs function to facilitate the conversions before comparison?
    # [YOUR CODE HERE]


    # Just dummy data to avoid errors
    #pred = np.zeros((10))
    # Use the model to predict labels for the test set (uncomment the line below)
    #pred = model.predict(X_test)

    # Translate predicted probabilities into pos/neg labels
    pred = translate_probs(pred)
    # Save the results for submission
    np.save('q1-pred.npy', pred)


if __name__ == '__main__':
	main()
