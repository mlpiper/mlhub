"""
This is a sample code showing how tensorflow code can be instrumented in MCenter.
"""
import argparse

import numpy as np
import tensorflow as tf
from parallelm.mlops import mlops as mlops
from parallelm.mlops.predefined_stats import PredefinedStats
from parallelm.mlops.stats.bar_graph import BarGraph

"""
Function to add the arguments that are provided as arguments to the component.
"""


def add_parameters(parser):
    parser.add_argument("--output_file", dest="output_file", type=str, required=False, default="tmp/image_predictions",
                        help='Prediction directory')
    parser.add_argument("--model_dir", dest="model_dir", type=str, required=False, help='Model Directory',
                        default="/tmp/tf_log")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    add_parameters(parser)
    args = parser.parse_args()

    # Initialize MLOps Library
    mlops.init()

    # Create synthetic data (Gaussian Distribution, Poisson Distribution and Beta Distribution)
    num_samples = 50
    num_features = 20

    np.random.seed(0)
    g = np.random.normal(0, 1, (num_samples, num_features))
    p = np.random.poisson(0.7, (num_samples, num_features))
    b = np.random.beta(2, 2, (num_samples, num_features))

    test_data = np.concatenate((g, p, b), axis=0)
    np.random.seed()
    features = test_data[np.random.choice(test_data.shape[0], num_samples, replace=False)]

    # Start tensorflow session
    sess = tf.InteractiveSession()
    tag_set = ["serve"]
    if args.model_dir is not None:
        try:
            print("args.model_dir = ", args.model_dir)
            tf.saved_model.loader.load(sess, tag_set, args.model_dir)
        except Exception as e:
            print("Model not found")
            print("Got exception: " + str(e))
            return 0

    # Output Health Statistics to MCenter
    # MLOps API to report the distribution statistics of each feature in the data and compare it automatically with the ones
    # reported during training to generate the similarity score.
    mlops.set_data_distribution_stat(data=features)

    # Output the number of samples being processed using MCenter
    mlops.set_stat(PredefinedStats.PREDICTIONS_COUNT, len(features))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("features:0")
    y_pred = graph.get_tensor_by_name("predictions:0")
    predictions = sess.run(y_pred, {x: features})
    print('predictions', np.array(predictions))

    # Ouput prediction distribution as a BarGraph using MCenter
    predict_int = np.argmax(predictions, axis=1)
    unique, counts = np.unique(predict_int, return_counts=True)
    counts = list(map(int, counts))
    x_series = list(map(str, unique))
    mlt = BarGraph().name("Prediction Distribution").cols(x_series).data(list(counts))
    mlops.set_stat(mlt)

    # Show average prediction probability value for each prediction
    num_labels = len(np.unique(predict_int))
    probability = np.zeros((num_labels,))
    for a in range(0, num_labels):
        temp = predictions[np.argmax(predictions, axis=1) == a, :]
        print(temp)
        probability[a] = np.mean(temp[:, a])
    print("probability", list(np.squeeze(probability)))

    # Plot average probability in each class using MCenter
    bg = BarGraph().name("Probability of Each Label").cols(x_series).data(list(np.squeeze(probability)))
    mlops.set_stat(bg)


if __name__ == "__main__":
    main()
