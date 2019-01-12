'''
@author: David Rubio Vallejo
Stat NLP - HW3

This module contains functions to process the JSON files that hold the train, dev, and test sets. When run as main it
saves as a pickled file the vector representation of each datapoint in each dataset using the spaCy language model. It
also pickles the dictionary mapping senses and types to the index each of their possible values occupies in the label
vector (size 21), so that it can be reused when creating the JSON objects that the corresponding neural network has to
ouput for the assignment to be evaluated properly.
'''


import json
import codecs
import spacy
import numpy as np
import pickle


def read_json():
    """Loads JSON objects containing the train, dev, and test datasets"""

    train_file = codecs.open('train/relations.json', encoding='utf8')
    dev_file = codecs.open('dev/relations.json', encoding='utf8')
    test_file = codecs.open('test/relations.json', encoding='utf8')

    train_set = [json.loads(x) for x in train_file]
    dev_set = [json.loads(x) for x in dev_file]
    test_set = [json.loads(x) for x in test_file]

    return train_set, dev_set, test_set


def process_data_collection(train_dev_test_tupl):
    """Takes a loaded JSON object in the shape of a triple (training, dev, and test sets) and returns a triple where
    each of those dataset have been translated into a double of matrices (first elem matrix with feature vectors of each
    doc, second elem matrix with true label vectors). Language model to create vectors is also loaded here."""

    # language model we'll be using
    langModel = spacy.load('en_vectors_web_lg')
    print("Language model loaded...")

    senses_set = get_senses(train_dev_test_tupl[0])
    # dict with sense labels mapping to an int to be used as the index they'll have in the true label vectors
    sense_index_dict = {v:k for k, v in enumerate(senses_set) }
    with open('sense_index_dict.pkl', 'wb') as f:
        pickle.dump(sense_index_dict, f)
    print("Index dict created and pickled...")

    training_set = create_vectors(train_dev_test_tupl[0], langModel, sense_index_dict)
    print("Training vectors created...")

    dev_set = create_vectors(train_dev_test_tupl[1], langModel, sense_index_dict)
    print("Dev vectors created...")

    test_set = create_vectors(train_dev_test_tupl[2], langModel, sense_index_dict)
    print("Test vector created...")

    return training_set, dev_set, test_set


def create_vectors(dataset, langModel, sense_index_dict):
    """Outputs a tupl where first elem is a matrix of feature vectors for each doc in the input dataset, and second
    elem is a matrix with the true labels for those docs"""

    # matrix that will store the vector representations of the documents in the dataset argument
    instance_vector_collection = np.empty((len(dataset), 900))
    # matrix that holds the true label vectors for each doc
    true_label_vector_collection = np.empty((len(dataset), len(sense_index_dict)))

    instance_nb = 0

    # for each document in the dataset, translate it into a vector
    for instance in dataset:
        # first elem of tupl is the doc vector, second elem is the true label vector for that doc
        vector_tupl = create_instance_vector(instance, langModel, sense_index_dict)

        instance_vector_collection[instance_nb] = vector_tupl[0]

        true_label_vector_collection[instance_nb] = vector_tupl[1]

        instance_nb += 1

    return instance_vector_collection, true_label_vector_collection


def create_instance_vector(instance, langModel, sense_index_dict):

    # each of the elements that we want to use to create the feature vector
    arg1 = instance['Arg1']['RawText']
    arg2 = instance['Arg2']['RawText']
    connective = instance['Connective']['RawText']

    concatd_vector = np.array([])

    # get each word from each of those elements, get its vector representation and add them together
    for elem in [arg1, arg2, connective]:
        # creates spaCy 'Document' object based on the language model
        doc = langModel(elem)
        # each vector in the langModel has 300 dimensions
        elem_vector = np.zeros(300)
        # if the connective is empty (as in the case of implicit datapoints, we wont enter the loop and an array of
        # zeros will be appended
        for word in doc:
            elem_vector = elem_vector + word.vector

        # the vector for each document will be the concatenation of the vectors for arg1, arg2, and connective
        # so each doc vector will have 900 dimensions because each word has 300 dimensions in the spaCy model used here
        # and I'm adding the vectors of each word in the cases of arg1 and arg2
        concatd_vector = np.concatenate((concatd_vector, elem_vector))

    true_label_vector = np.zeros(len(sense_index_dict))
    # get the index that the sense will have in the true label vector. 'Sense' key in the JSON object maps to a list!
    for sense in instance["Sense"]:
        sense_index = sense_index_dict[sense]
        true_label_vector[sense_index] = 1

    return concatd_vector, true_label_vector


def get_senses(train_set_json):
    """Passes through the training set and compiles a set of the senses available"""
    # the different senses and types of the connectives
    senses_set = set()

    for elem in train_set_json:
        for sense in elem['Sense']:
            if sense not in senses_set:
                senses_set.add(sense)

    return senses_set

if __name__ == '__main__':


    train_dev_test_sets = read_json()
    print("JSON objects loaded...")

    train_dev_test_vectors = process_data_collection(train_dev_test_sets)

    # Commented out not to overwrite by mistake
    # with open('train_vectors.pkl', 'wb') as f:
    #     pickle.dump(train_dev_test_vectors[0], f)
    # with open('dev_vectors.pkl', 'wb') as f:
    #     pickle.dump(train_dev_test_vectors[1], f)
    # with open('test_vectors.pkl', 'wb') as f:
    #     pickle.dump(train_dev_test_vectors[2], f)





