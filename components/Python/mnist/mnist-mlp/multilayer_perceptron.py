""" Multilayer Perceptron.
A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------
# This file was modified from its original form to include statistical reporting
# and to facilitate testing. These code modifications are provided "as is" without
# warranty of any kind.
# ==============================================================================

from __future__ import print_function

import argparse

import tensorflow as tf

## MLOps start
from parallelm.mlops import mlops
from parallelm.mlops.stats_category import StatCategory
## MLOps end

# Parameters
learning_rate = 0.001
batch_size = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


# Create the model
def multilayer_perceptron(x):

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Train the model
def train(mnist_data, training_epochs, model_dir, display_step):

    # tf Graph input
    sample = tf.placeholder("float", [None, n_input])
    X = tf.identity(sample, name='x') # use tf.identity() to assign name
    Y = tf.placeholder("float", [None, n_classes])

    # Construct model
    logits = multilayer_perceptron(X)
    inference = tf.nn.softmax(logits, name='y')  # Apply softmax to logits

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # pred = tf.nn.softmax(logits, name='y')  # Apply softmax to logits

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        print("Start training")
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist_data.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist_data.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch 

            # Display stats periodically
            if (epoch + 1) % display_step == 0 or (epoch + 1) == training_epochs:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

                # Test model
                correct_prediction = tf.equal(tf.argmax(inference, 1), tf.argmax(Y, 1))

                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                accuracy = accuracy.eval({X: mnist_data.test.images, Y: mnist_data.test.labels})
                print("Accuracy:", accuracy)

                # MLOps start
                mlops.set_stat("epochs", epoch)
                mlops.set_stat("Accuracy", accuracy * 100)
                # multiply by 1 to convert into double until mlops supports float32
                mlops.set_stat("Cross entropy", avg_cost * 1)
                # MLOps end

        print("Optimization Finished!")

        # Save the trained model
        print('Exporting trained model to', model_dir)
        builder = tf.saved_model.builder.SavedModelBuilder(model_dir)

        tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(inference)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'inputs': tensor_info_x},
              outputs={'outputs': tensor_info_y},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
              'predict_images': prediction_signature,
              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,
          },
          legacy_init_op=legacy_init_op)

        builder.save()

        print('Done exporting!')


def get_input(input_dir):
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets(input_dir, one_hot=True)

def add_parameters(parser):

    parser.add_argument("--input_dir",
                        dest="input_dir",
                        type=str,
                        required=False,
                        help='Directory for caching input data',
                        default="/tmp/mnist_data")

    parser.add_argument("--model_dir", dest="model_dir", help='Where to output model', type=str, required=True)

    # Stats arguments
    parser.add_argument("--display_step", dest="display_step", type=int, default=1, required=False)

    parser.add_argument("--epochs", dest="epochs", help='Training epochs', type=int, default=15, required=False)


def main():

    parser = argparse.ArgumentParser()
    add_parameters(parser)
    args = parser.parse_args()

    mnist_data = get_input(args.input_dir)

    X = mnist_data.train.images

    ## MLOps start
    # Initialize the mlops library
    mlops.init()

    # Report the feature distribution for the training data
    mlops.set_data_distribution_stat(X)
    ## MLOps end

    train(mnist_data, args.epochs, args.model_dir, args.display_step)

    ## MLOps start
    # Release mlops resources
    mlops.done()
    ## MLOps end

if __name__ == "__main__":
    main()
