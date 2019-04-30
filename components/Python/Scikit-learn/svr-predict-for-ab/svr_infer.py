from __future__ import print_function

import argparse
import pickle

import numpy as np
from parallelm.mlops import StatCategory as st
from parallelm.mlops import mlops as mlops
from parallelm.mlops.predefined_stats import PredefinedStats
from parallelm.mlops.stats.bar_graph import BarGraph
from sklearn.datasets import make_regression


def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="# samples")
    parser.add_argument("--num_features", help="# features")

    parser.add_argument("--threshold", help="MAE threshold")

    parser.add_argument("--input-model", help="Path of input model to create")
    options = parser.parse_args()
    return options


def main():
    pm_options = parse_args()
    # Initialize MLOps Library
    mlops.init()
    # Load the model
    if pm_options.input_model is not None:
        try:
            filename = pm_options.input_model
            file_obj = open(filename, 'rb')
            mlops.set_stat("model_file", 1)
        except Exception as e:
            print("Model not found")
            print("Got exception: {}".format(e))
            mlops.set_stat("model_file", 0)
            mlops.done()
            return 0

    regression = pickle.load(file_obj)

    # Create synthetic data (Gaussian Distribution, Poisson Distribution and Beta Distribution)
    num_samples = int(pm_options.num_samples)
    num_features = int(pm_options.num_features)

    mae_threshold = float(pm_options.threshold)

    # Create synthetic data using scikit learn
    X, y = make_regression(n_samples=num_samples,
                           n_features=num_features,
                           n_informative=2,
                           random_state=42)

    # for making labels all positive
    y = y + -1 * np.min(y)

    # Separate into features and labels
    features = X
    labels = y

    # Add noise to the data
    noisy_features = np.random.uniform(0, 10) * \
                     np.random.normal(0, 1,
                                      (num_samples, num_features))
    features = features + noisy_features

    # Output Health Statistics to MCenter
    # MLOps API to report the distribution statistics of each feature in the data and compare it automatically with the ones
    # reported during training to generate the similarity score.
    mlops.set_data_distribution_stat(features)

    # Output the number of samples being processed using MCenter
    mlops.set_stat(PredefinedStats.PREDICTIONS_COUNT, num_samples, st.TIME_SERIES)

    # Predict labels
    labels_pred = regression.predict(features)

    hist_pred, bin_edges_pred = np.histogram(labels_pred)

    # Output prediction label distribution as a BarGraph using MCenter
    pred_label_bar = BarGraph().name("User Defined: Prediction Label Distribution") \
        .cols(bin_edges_pred.astype(str).tolist()) \
        .data(hist_pred.tolist()) \
        .as_continuous()

    mlops.set_stat(pred_label_bar)

    ##########################################################################
    #################### Start: Output Sample/Conversions ####################
    ########################################################################@@
    mae = np.absolute(labels_pred - labels)
    conversions = sum(i < mae_threshold for i in mae)
    samples = num_samples

    mlops.set_stat("samples", samples)

    mlops.set_stat("conversions", conversions)

    ########################################################################
    #################### End: Output Sample/Conversions ####################
    ########################################################################

    # Terminate MLOPs
    mlops.done()


if __name__ == "__main__":
    main()
