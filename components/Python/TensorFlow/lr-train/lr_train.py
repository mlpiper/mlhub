"""
This is sample code showing how tensorflow code can be instrumented in MCenter.
"""
import argparse
import time

import numpy as np
import tensorflow as tf

from sklearn.datasets import make_classification
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.graph import Graph
from parallelm.mlops.stats.graph import MultiGraph
from parallelm.mlops.stats.multi_line_graph import MultiLineGraph

"""
Function to add the arguments that are provided as arguments to the component.
"""


def add_parameters(parser):
    parser.add_argument("--step_size", dest="step_size", type=float, required=False, default=0.01, help='Learning rate')
    parser.add_argument("--iterations", dest="iterations", type=int, required=False, default=100,
                        help='Number of training iterations')
    parser.add_argument("--model_version", dest="model_version", type=int, required=False, default=1,
                        help='Model version')
    parser.add_argument("--stats_interval", dest="stats_interval", type=int, required=False, default=1,
                        help='Print stats after this number of iterations')
    parser.add_argument("--save_dir", dest="save_dir", type=str, required=False,
                        help='Directory for saving the trained model', default="/tmp/tf_model")
    parser.add_argument("--text_model_format", dest="use_text", required=False, default=False, action='store_true',
                        help='Whether SavedModel should be binary or text')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    add_parameters(parser)
    args = parser.parse_args()
    print("PM: Configuration:")
    print("PM: Step size:                  [{}]".format(args.step_size))
    print("PM: Iterations:                 [{}]".format(args.iterations))
    print("PM: Model version:              [{}]".format(args.model_version))
    print("PM: Stats interval:             [{}]".format(args.stats_interval))
    print("PM: Save dir:                   [{}]".format(args.save_dir))

    # Initialize MLOps Library
    mlops.init()

    # print the number of iteration used by optimization algorithm
    print('Training for %i iterations' % args.iterations)

    # Create sythetic data using scikit learn
    num_samples = 50
    num_features = 20

    features, labels = make_classification(n_samples=50, n_features=20, n_informative=2, n_redundant=1, n_classes=3,
                                           n_clusters_per_class=1, random_state=42)

    # Add noise to the data
    noisy_features = np.random.uniform(0, 5) * np.random.normal(0, 1, (num_samples, num_features))
    features = features + noisy_features

    num_features = (features.shape[1])
    num_labels = len(np.unique(labels))

    # One-hot encode labels for all data
    onehot_labels = np.eye(num_labels)[labels]

    # Label distribution in training
    value, counts = np.unique(labels, return_counts=True)
    label_distribution = np.asarray((value, counts)).T
    column_names = value.astype(str).tolist()
    print("Label distributions: \n {0}".format(label_distribution))

    # Output label distribution as a BarGraph using MCenter
    bar = BarGraph().name("Label Distribution").cols((label_distribution[:, 0]).astype(str).tolist()).data(
        (label_distribution[:, 1]).tolist())
    mlops.set_stat(bar)

    # Output Health Statistics to MCenter
    # Report features whose distribution should be compared during inference
    mlops.set_data_distribution_stat(features)

    # Algorithm parameters parsed from arguments
    learning_rate = args.step_size
    training_epochs = args.iterations
    display_step = args.stats_interval

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, num_features], name="features")
    y = tf.placeholder(tf.float32, [None, num_labels], name="labels")

    # Set model weights
    W = tf.Variable(tf.zeros([num_features, num_labels]))
    b = tf.Variable(tf.zeros([num_labels]))

    # Store values for saving model
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b, name="predictions")  # Softmax

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # Start timer
    training_start_time = time.time()

    # Initialize the variables in a tf session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    iteration_array = []
    cost_array = []
    accuracy_array = []

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        temp, c, a = sess.run([optimizer, cost, accuracy], feed_dict={x: features, y: onehot_labels})
        # Compute average loss
        avg_cost += c / num_samples
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            iteration_array.append(epoch)
            cost_array.append(avg_cost)
            accuracy_array.append(np.float(a))
            print("accuracy", a)
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    # Plot the cost function using MCenter
    gg = Graph().name("Cost function across epochs").set_x_series(iteration_array).add_y_series(
        label="Cost Function Across Iterations", data=cost_array)
    gg.x_title("Average Cost")
    gg.y_title('Iterations')
    mlops.set_stat(gg)

    # Plot the accuracy function using MCenter
    gg1 = Graph().name("Accuracy across epochs").set_x_series(iteration_array).add_y_series(
        label="Accuracy Across Iterations", data=accuracy_array)
    gg1.x_title("Accuracy")
    gg1.y_title('Iterations')
    mlops.set_stat(gg1)

    # Plot accuracy and cost across epochs using MCenter
    mg = MultiGraph().name("Cost and Accuracy Progress Across Epochs")
    mg.add_series(x=iteration_array, label="Cost Function Across Iterations", y=cost_array)
    mg.add_series(x=iteration_array, label="Accuracy across epochs", y=accuracy_array)
    mlops.set_stat(mg)

    # Plot final cost and accuracy in this session using MCenter
    mlt = MultiLineGraph().name("Final Accuracy and Cost").labels(["Cost", "Accuracy"])
    mlt.data([cost_array[-1], accuracy_array[-1]])
    mlops.set_stat(mlt)

    # Save the model
    export_path = args.save_dir
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    values, indices = tf.nn.top_k(y, num_labels)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(num_labels)]))
    prediction_classes = table.lookup(tf.to_int64(indices))

    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs},
            outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
                     tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: classification_outputs_scores},
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

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
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save(as_text=args.use_text)


if __name__ == "__main__":
    main()
