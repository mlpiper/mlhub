import argparse

import numpy as np
import sklearn
from parallelm.mlops import mlops as mlops
# use below import if user wants to user RegressionMetrics predefined metrics names.
from parallelm.mlops.metrics_constants import RegressionMetrics
from parallelm.mlops.stats.bar_graph import BarGraph
from sklearn.datasets import make_regression
from sklearn.svm import SVR


def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="# samples")
    parser.add_argument("--num_features", help="# features")

    parser.add_argument("--kernel", help="Kernel")
    parser.add_argument("--degree", help="Degree")
    parser.add_argument("--gamma", help="Gamma")
    parser.add_argument("--tol", help="Tol")
    parser.add_argument("--max_iter", dest="max_iter", type=int, required=False, default=100,
                        help='Maximum number of iterations')
    parser.add_argument("--output-model", help="Data file to save model")
    options = parser.parse_args()
    return options


def main():
    pm_options = parse_args()

    print("PM: Configuration:")
    print("PM: # Sample:                    [{}]".format(pm_options.num_samples))
    print("PM: # Features:                  [{}]".format(pm_options.num_features))

    print("PM: Kernel:                      [{}]".format(pm_options.kernel))
    print("PM: Degree:                      [{}]".format(pm_options.degree))
    print("PM: Gamma:                       [{}]".format(pm_options.gamma))
    print("PM: Tolerance:                   [{}]".format(pm_options.tol))
    print("PM: Maximum iterations:          [{}]".format(pm_options.max_iter))

    print("PM: Output model:                [{}]".format(pm_options.output_model))

    # Initialize MLOps Library
    mlops.init()

    num_samples = int(pm_options.num_samples)
    num_features = int(pm_options.num_features)

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
    # MLOps API to report the distribution statistics of each feature in the data
    mlops.set_data_distribution_stat(features)

    hist, bin_edges = np.histogram(labels)

    # Output label distribution as a BarGraph using MCenter
    bar = BarGraph().name("User Defined: Label Distribution") \
        .cols((bin_edges).astype(str).tolist()) \
        .data((hist).tolist()) \
        .as_continuous()

    mlops.set_stat(bar)

    # Create a model that should be deployed into production
    final_model = SVR(kernel=pm_options.kernel,
                      degree=int(pm_options.degree),
                      gamma=str(pm_options.gamma),
                      tol=float(pm_options.tol),
                      max_iter=int(pm_options.max_iter))

    final_model.fit(features, labels)

    labels_pred = final_model.predict(features)
    hist_pred, bin_edges_pred = np.histogram(labels_pred)

    # Output prediction label distribution as a BarGraph using MCenter
    pred_label_bar = BarGraph().name("User Defined: Prediction Label Distribution") \
        .cols((bin_edges_pred).astype(str).tolist()) \
        .data((hist_pred).tolist()) \
        .as_continuous()

    mlops.set_stat(pred_label_bar)

    y_true = labels
    y_pred = labels_pred

    # Regression Metrics
    ##########################################################################
    #################### Start: Output Explained Variance ####################
    ##########################################################################

    evs = sklearn.metrics.explained_variance_score(y_true, y_pred)

    #################### OLD WAY ####################
    # First Way
    # mlops.set_stat("User Defined: Explained Variance ", evs)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(RegressionMetrics.EXPLAINED_VARIANCE_SCORE, evs)

    # OR

    # Third Way
    mlops.metrics.explained_variance_score(y_true=labels, y_pred=labels_pred)
    #################### DONE NEW WAY ####################

    ########################################################################
    #################### End: Output Explained Variance ####################
    ########################################################################

    ######################################################################
    #################### Start: Output Mean Abs Error ####################
    ######################################################################

    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)

    #################### OLD WAY ####################
    # First Way
    # mlops.set_stat("User Defined: Mean Abs Error", mae)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(RegressionMetrics.MEAN_ABSOLUTE_ERROR, mae)

    # OR

    # Third Way
    mlops.metrics.mean_absolute_error(y_true=labels, y_pred=labels_pred)
    #################### DONE NEW WAY ####################

    ####################################################################
    #################### End: Output Mean Abs Error ####################
    ####################################################################

    ##########################################################################
    #################### Start: Output Mean Squared Error ####################
    ##########################################################################

    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)

    #################### OLD WAY ####################
    # First Way
    # mlops.set_stat("User Defined: Mean Squared Error", mse)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(RegressionMetrics.MEAN_SQUARED_ERROR, mse)

    # OR

    # Third Way
    mlops.metrics.mean_squared_error(y_true=labels, y_pred=labels_pred)
    #################### DONE NEW WAY ####################

    ########################################################################
    #################### End: Output Mean Squared Error ####################
    ########################################################################

    ##############################################################################
    #################### Start: Output Mean Squared Log Error ####################
    ##############################################################################

    msle = sklearn.metrics.mean_squared_log_error(y_true, y_pred)

    #################### OLD WAY ####################
    # First Way
    # mlops.set_stat("User Defined: Mean Squared Log Error", msle)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(RegressionMetrics.MEAN_SQUARED_LOG_ERROR, msle)

    # OR

    # Third Way
    mlops.metrics.mean_squared_log_error(y_true=labels, y_pred=labels_pred)

    #################### DONE NEW WAY ####################

    ############################################################################
    #################### End: Output Mean Squared Log Error ####################
    ############################################################################

    ########################################################################
    #################### Start: Output Median Abs Error ####################
    ########################################################################

    median_ae = sklearn.metrics.median_absolute_error(y_true, y_pred)

    #################### OLD WAY ####################
    # First Way
    # mlops.set_stat("User Defined: Median Abs Error", median_ae)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(RegressionMetrics.MEDIAN_ABSOLUTE_ERROR, median_ae)

    # OR

    # Third Way
    mlops.metrics.median_absolute_error(y_true=labels, y_pred=labels_pred)
    #################### DONE NEW WAY ####################

    ######################################################################
    #################### End: Output Median Abs Error ####################
    ######################################################################

    ################################################################
    #################### Start: Output R2 Score ####################
    ################################################################

    r2_s = sklearn.metrics.r2_score(y_true, y_pred)

    #################### OLD WAY ####################
    # First Way
    # mlops.set_stat("User Defined: R2 Score", r2_s)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(RegressionMetrics.R2_SCORE, r2_s)

    # OR

    # Third Way
    mlops.metrics.r2_score(y_true=labels, y_pred=labels_pred)
    #################### DONE NEW WAY ####################

    ##############################################################
    #################### End: Output R2 Score ####################
    ##############################################################

    # Save the model
    import pickle
    model_file = open(pm_options.output_model, 'wb')
    pickle.dump(final_model, model_file)
    model_file.close()
    # Terminate MLOPs
    mlops.done()


if __name__ == "__main__":
    main()
