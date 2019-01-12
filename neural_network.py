'''
@author: David Rubio Vallejo
Stat NLP - HW3
'''

import tensorflow as tf
import json
import pickle
from preprocessing import read_json
import os

# number of iterations to train the model
ITERATIONS = 300


class NeuralNetwork:

    def __init__(self, train, dev, test):
        # training set
        self.training_vectors = train[0]
        self.training_labels = train[1]
        # devset
        self.dev_vectors = dev[0]
        self.dev_labels = dev[1]
        # test set
        self.test_vectors = test[0]
        self.test_labels = test[1]

        # first layer: rows = nb of feats in each input doc (here it should be 900); cols = nb of neurons in layer
        self.W1 = tf.Variable(tf.random_uniform((len(self.training_vectors[0]), 600), -1, 1))
        # self.b1 = tf.Variable(tf.zeros(1,))
        self.b1 = tf.Variable(tf.random_uniform((1,), -1, 1))

        # second layer
        self.W2 = tf.Variable(tf.random_uniform((600, 300), -1, 1))
        # self.b2 = tf.Variable(tf.zeros(1,))
        self.b2 = tf.Variable(tf.random_uniform((1,), -1, 1))

        # third layer
        self.W3 = tf.Variable(tf.random_uniform((300, 100), -1, 1))
        # self.b3 = tf.Variable(tf.zeros(1,))
        self.b3 = tf.Variable(tf.random_uniform((1,), -1, 1))

        # output layer
        self.W4 = tf.Variable(tf.random_uniform((100, len(self.training_labels[0])), -1, 1))
        # self.b3 = tf.Variable(tf.zeros(1,))
        self.b4 = tf.Variable(tf.random_uniform((1,), -1, 1))

        # the matrix that represents the input data
        # can use [None, 900] for any row size input
        self.x = tf.placeholder(tf.float32, [None, 900])

        # specific configuration of the NN
        self.predicted = tf.nn.softmax(
                            tf.matmul(
                               tf.nn.sigmoid(
                                  tf.matmul(
                                       tf.nn.sigmoid(
                                           tf.matmul(
                                                tf.nn.sigmoid(
                                                    tf.matmul(self.x, self.W1)
                                                    + self.b1),
                                                self.W2)
                                           + self.b2),
                                       self.W3)
                                  + self.b3),
                               self.W4)
                            + self.b4)

    def train_nn(self):
        """Trains the neural network object with minibatch and the specified hyperparameters"""

        # creates the matrix that will contain the gold labels to obtain the cross entropy loss
        gold_labels = tf.placeholder(tf.float32, [None, len(self.training_labels[0])])

        # Loss: cross entropy
        cross_entropy = - tf.reduce_sum(gold_labels * tf.log(self.predicted), axis=1)

        # Regularizer: using L2
        reg_lambda = 0.0001
        regularizers = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W3) + tf.nn.l2_loss(self.W4)

        cross_entropy = tf.reduce_mean(cross_entropy + reg_lambda * regularizers)

        # Optimization: using Adam
        learning_rate = 0.001
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy)

        # Initializing session
        sess = tf.Session()
        # Initialization of the placeholder variables
        sess.run(tf.global_variables_initializer())

        for iteration in range(ITERATIONS):
            # Implementation with minibatch (size = 135). 241 is a divisor of 32535 (the length of training set)
            minibatch_size = len(self.training_vectors) // 241
            for i in range(minibatch_size):
                # do one run over the NN with the datapoints fitting in one minbatch
                sess.run(
                    train_step,
                    feed_dict={
                        self.x: self.training_vectors[i * minibatch_size: (i + 1) * minibatch_size],
                        gold_labels: self.training_labels[i * minibatch_size: (i + 1) * minibatch_size]})

            # at every iteration multiple of 10, calculate the loss on devset and training set, and print them
            if iteration % 10 == 0:
                dev_loss = tf.reduce_sum(cross_entropy)
                dev_loss_value = sess.run(dev_loss,
                                          feed_dict={self.x: self.dev_vectors,
                                                     gold_labels: self.dev_labels})
                train_loss = tf.reduce_sum(cross_entropy)
                train_loss_value = sess.run(train_loss,
                                            feed_dict={self.x: self.training_vectors,
                                                       gold_labels: self.training_labels})

                print('iter:', iteration, ', loss on train set:', train_loss_value, ', loss on devset:', dev_loss_value)

        # saves the model in specified location
        saver = tf.train.Saver()
        save_path = saver.save(sess, os.path.join(os.getcwd(), "Trained_model", "discourse_model.tfw"))
        print("Model saved in path: %s" % save_path)

    def create_predicted_jsons(self, index_sense_dict):
        """Runs the model on the test set and creates a json object for each datapoint that contains the 'sense' label
        predicted. The output is a list containing those json objects"""

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Restores the saved trained model.
            saver.restore(sess, os.path.join(os.getcwd(), "Trained_model", "discourse_model.tfw"))

            # The predicted vector labels for each test document
            y_hats = sess.run(self.predicted, feed_dict={self.x: self.test_vectors})
            # Applies hardmax to each predicted vector
            y_hats = sess.run(tf.contrib.seq2seq.hardmax(y_hats))
            # Method call to create the json objects
            output_json_list = self.translate_into_json(y_hats, index_sense_dict)

            return output_json_list

    def translate_into_json(self, results, sense_index_dict):
        """Input: predicted vector labels for the test documents with hardmax applied to them (together with the
        dictionary mapping each index to a sense that was created in the 'preprocessing.py' file)."""

        # creates a reverse of the input dict so that we can access the sense name given the index of the 1 in the
        # one-hot-vectors that are the predicted labels for the test-set datapoints
        index_sense_dict = {index: sense for sense, index in sense_index_dict.items()}

        # list containing the json objects corresponding to the test datapoints
        test_json_objects = read_json()[2]

        output_json_list = []

        # loop over each predicted label for the test set
        for k in range(len(results)):
            # get each instance and reset the sense value
            instance = test_json_objects[k]

            new_json = {"Arg1": {"TokenList": []},
                        "Arg2": {"TokenList": []},
                        "Connective": {"TokenList": []},
                        "DocID": instance["DocID"],
                        "Sense": [],
                        "Type": instance["Type"]}

            # adds the span offset of both arguments
            new_json["Arg1"]["TokenList"].append(instance["Arg1"]["CharacterSpanList"][0])
            new_json["Arg2"]["TokenList"].append(instance["Arg2"]["CharacterSpanList"][0])

            # loop over each value in the predicted label vector for the specific instance
            for i in range(len(results[k])):
                # if the value at this index is 1, look for the the sense/type that corresponds to that index and
                # add it to the JSON representation of the object
                if results[k][i] == 1:
                    sense = index_sense_dict[i]
                    new_json["Sense"].append(sense)

            output_json_list.append(new_json)

        return output_json_list


if __name__ == '__main__':

    # Loads the pickled objects and stores them into variables
    training_vectors_and_labels = None
    dev_vectors_and_labels = None
    test_vectors_and_labels = None
    sense_index_dict = None

    with open('train_vectors.pkl', 'rb') as f:
        training_vectors_and_labels = pickle.load(f)
    with open('dev_vectors.pkl', 'rb') as f:
        dev_vectors_and_labels = pickle.load(f)
    with open('test_vectors.pkl', 'rb') as f:
        test_vectors_and_labels = pickle.load(f)
    with open('sense_index_dict.pkl', 'rb') as f:
        sense_index_dict = pickle.load(f)

    # creates Neural Network object
    nn = NeuralNetwork(training_vectors_and_labels, dev_vectors_and_labels, test_vectors_and_labels)
    # trains NN and saves the trained model into a file
    nn.train_nn()

    # creates the list of predicted json objects and saves it into a file
    predicted_jsons = nn.create_predicted_jsons(sense_index_dict)
    with open('output.json', 'w') as f:
        for single_json in predicted_jsons:
            print(json.dumps(single_json), file=f)
