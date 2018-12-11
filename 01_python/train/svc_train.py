import argparse

import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from parallelm.mlops import StatCategory as st
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph


def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="# samples")
    parser.add_argument("--num_features", help="# features")
    parser.add_argument("--C", help="C Parameter")
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

    print("PM: C:                           [{}]".format(pm_options.C))
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
    X, y = make_classification(n_samples=num_samples,
                               n_features=num_features,
                               n_informative=2,
                               n_redundant=1,
                               n_classes=3,
                               n_clusters_per_class=1,
                               random_state=42)

    # Separate into features and labels
    features = X
    labels = y

    # Add noise to the data
    noisy_features = np.random.uniform(0, 10) * \
                     np.random.normal(0, 1,
                                      (num_samples, num_features))
    features = features + noisy_features

    # Create a model that should be deployed into production
    final_model = SVC(C=float(pm_options.C),
                      kernel=pm_options.kernel,
                      degree=int(pm_options.degree),
                      gamma=str(pm_options.gamma),
                      tol=float(pm_options.tol),
                      max_iter=int(pm_options.max_iter))

    final_model.fit(features, labels)

    # Accuracy for the chosen model
    accuracy = final_model.score(features, labels)
    print("Accuracy values: \n {0}".format(accuracy))

    # Label distribution in training
    value, counts = np.unique(labels, return_counts=True)
    label_distribution = np.asarray((value, counts)).T
    column_names = value.astype(str).tolist()
    print("Label distributions: \n {0}".format(label_distribution))

    # Output label distribution as a BarGraph using MCenter
    bar = BarGraph().name("Label Distribution").cols((label_distribution[:, 0]).astype(str).tolist()).data(
        (label_distribution[:, 1]).tolist())
    mlops.set_stat(bar)

    # Output accuracy of the chosen model using MCenter
    mlops.set_stat("Accuracy", accuracy, st.TIME_SERIES)

    # Output Health Statistics to MCenter
    # MLOps API to report the distribution statistics of each feature in the data
    mlops.set_data_distribution_stat(features)

    # Save the model
    import pickle
    model_file = open(pm_options.output_model, 'wb')
    pickle.dump(final_model, model_file)
    model_file.close()
    # Terminate MLOPs
    mlops.done()


if __name__ == "__main__":
    main()
