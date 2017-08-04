import os

import tensorflow as tf

# For restoring pickled data
import pickle

# Suffle data set
import random

import numpy as np

# Directory containing extracted CROHME data
data_dir = 'data'

# Define the size of a square input image
box_size = 32
# Store the total number of classes our classifier needs to learn
# num_classes = 10

classes = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '(', ')', '[', ']', '+', '-', '/', '=', '\geq', '\gt', '\leq', '\lt', '\\neq', '\\times', '\div']
num_classes = len(classes)

def load_data(path_to_data):

    with open(path_to_data, 'rb') as data:

        print('Restoring data set ...')
        data_set = pickle.load(data)

    return data_set

def to_one_hot(label_string):

    one_hot = np.zeros(shape=(num_classes), dtype=np.int8)

    one_hot[classes.index(label_string)] = 1

    return one_hot

def next_batch(data_set, batch_size):

    # Shuffle data set
    random.shuffle(data_set)

    random_idx = random.randint(0, len(data_set) - batch_size)

    batch = data_set[random_idx: random_idx + batch_size]

    patterns = []
    labels = []
    for entry in batch:

        # Flatten pattern array (2D ---> 1D)
        flattened = entry['pattern'].flatten()

        # We convert string label to 'one-hot' format
        one_hot = to_one_hot(entry['label'])

        patterns.append(flattened)
        labels.append(one_hot)

    return patterns, labels

def train():

    # Load training & testing sets
    train_path = os.path.join(data_dir, 'train', 'train.pickle')
    test_path = os.path.join(data_dir, 'test', 'test.pickle')

    train_set = load_data(train_path)
    test_set = load_data(test_path)


    # Tensorflow placeholders will contain external inputs
    # Placeholder for input image features
    x = tf.placeholder(tf.float32, [None, box_size * box_size])
    # Placeholder for input image's one-hot encoded label
    y = tf.placeholder(tf.float32, [None, num_classes])


    # Hyperparameters

    # Number of nodes of hidden layer
    # Optimal value for number of nodes of hidden layer is a number
    # Between number of input features and number of classes
    num_nodes = int((box_size * box_size - num_classes) / 2)
    # Size of randomly selected batch of training & test sets
    batch_size = 15000

    learning_rate_alpha = 0.01
    numb_epoches = 1500
    # Display model accuracy every display_step epoches
    display_step = 10

    # Optimal Standard deviation for weights & biases can be calculated using the following formula:
    # sqrt(2) / sqrt(d), where d is the number of inputs to a given neuron.
    stddev = np.sqrt(2) / box_size
    # Standard deviation for weights & biases connecting output layer
    stddev_out = np.sqrt(2) / np.sqrt(num_nodes)

    # Neural network weights and biases
    # Hidden layer
    W = tf.Variable(tf.random_normal(shape=[box_size * box_size, num_nodes], stddev=stddev))
    b = tf.Variable(tf.random_normal(shape=[num_nodes], stddev=stddev))
    # Weights & biases connecting output layer
    W_out = tf.Variable(tf.random_normal(shape=[num_nodes, num_classes], stddev=stddev_out))
    b_out = tf.Variable(tf.random_normal(shape=[num_classes], stddev=stddev_out))


    # Define a learning model
    l1 = tf.nn.relu(tf.matmul(x, W) + b)
    pred = tf.nn.softmax(tf.matmul(l1, W_out) + b_out)


    # cross-entropy function will calculate loss value
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

    # Initialize optimizer that will lower loss value and improve performance of our model
    optimizer = tf.train.GradientDescentOptimizer(learning_rate_alpha).minimize(cross_entropy)


    # Compare predicted label (class) with input image's label
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate mean of ...
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # Run computational graph
    sess = tf.Session()

    # To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Train model
    for epoch in range(numb_epoches):

        # Get random batch of traing set
        train_X, train_y = next_batch(train_set, batch_size)

        # Optimize model's weights and biases
        sess.run(optimizer, feed_dict={x: train_X, y: train_y})

        # Print accuracy - model's performance on test set
        test_X, test_y = next_batch(test_set, int(batch_size / 2))

        if epoch % display_step == 0:

            print(sess.run(accuracy, feed_dict={x: test_X, y: test_y}))

train()
