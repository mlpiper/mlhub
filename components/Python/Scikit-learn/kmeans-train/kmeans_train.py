import argparse

import numpy as np
import sklearn
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification


def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="# samples")
    parser.add_argument("--num_features", help="# features")
    parser.add_argument("--num_cluster", help="# Cluster")

    parser.add_argument("--init", help="Init")
    parser.add_argument("--n_init", help="N Init")
    parser.add_argument("--tol", help="Tol")
    parser.add_argument("--max_iter", dest="max_iter", type=int, required=False, default=100,
                        help='Maximum number of iterations')
    parser.add_argument("--precompute_distances", help="Pre-Compute Distances")
    parser.add_argument("--algorithm", help="Algorithm")

    parser.add_argument("--output-model", help="Data file to save model")
    options = parser.parse_args()
    return options


def main():
    pm_options = parse_args()

    print("PM: Configuration:")
    print("PM: # Sample:                    [{}]".format(pm_options.num_samples))
    print("PM: # Features:                  [{}]".format(pm_options.num_features))
    print("PM: # Classes:                   [{}]".format(pm_options.num_cluster))

    print("PM: Init:                        [{}]".format(pm_options.init))
    print("PM: N Init:                      [{}]".format(pm_options.n_init))
    print("PM: Tolerance:                   [{}]".format(pm_options.tol))
    print("PM: Maximum Iterations:          [{}]".format(pm_options.max_iter))
    print("PM: Pre-Compute Distances:       [{}]".format(pm_options.precompute_distances))
    print("PM: Algorithm:                   [{}]".format(pm_options.algorithm))

    print("PM: Output model:                [{}]".format(pm_options.output_model))

    # Initialize MLOps Library
    mlops.init()

    n_samples = int(pm_options.num_samples)
    n_features = int(pm_options.num_features)
    n_clusters = int(pm_options.num_cluster)

    init = str(pm_options.init)
    n_init = int(pm_options.n_init)
    max_iter = int(pm_options.max_iter)
    tol = float(pm_options.tol)
    precompute_distances = str(pm_options.precompute_distances)
    algorithm = str(pm_options.algorithm)
    verbose = 0
    n_jobs = 1

    # Create synthetic data using scikit learn
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=10,
                               n_redundant=1,
                               n_classes=n_clusters,
                               n_clusters_per_class=1,
                               random_state=42)

    # Separate into features and labels
    features = X
    labels_true = y

    # Add noise to the data
    noisy_features = np.random.uniform(0, 10) * \
                     np.random.normal(0, 1,
                                      (n_samples, n_features))
    features = features + noisy_features

    kmeans_model = KMeans(n_clusters=n_clusters,
                          init=init,
                          n_init=n_init,
                          max_iter=max_iter,
                          tol=tol,
                          precompute_distances=precompute_distances,
                          verbose=verbose,
                          random_state=None,
                          copy_x=True,
                          n_jobs=n_jobs,
                          algorithm=algorithm).fit(features, labels_true)

    mlops.set_stat("User Defined: Training Inertia", kmeans_model.inertia_)
    mlops.set_stat("User Defined: Training Iteration", kmeans_model.n_iter_)

    value, counts = np.unique(labels_true, return_counts=True)
    label_distribution = np.asarray((value, counts)).T

    # Output actual label distribution as a BarGraph using MCenter
    bar_true = BarGraph().name("User Defined: Actual Label Distribution") \
        .cols((label_distribution[:, 0]).astype(str).tolist()) \
        .data((label_distribution[:, 1]).tolist())
    mlops.set_stat(bar_true)

    # prediction labels
    labels_pred = kmeans_model.predict(features)

    value_pred, counts_pred = np.unique(labels_pred, return_counts=True)
    label_distribution_pred = np.asarray((value_pred, counts_pred)).T

    # Output prediction label distribution as a BarGraph using MCenter
    bar_pred = BarGraph().name("User Defined: Prediction Label Distribution") \
        .cols((label_distribution_pred[:, 0]).astype(str).tolist()) \
        .data((label_distribution_pred[:, 1]).tolist())
    mlops.set_stat(bar_pred)

    # Output Health Statistics to MCenter
    # MLOps API to report the distribution statistics of each feature in the data
    mlops.set_data_distribution_stat(features)

    ###########################################################################
    #################### Start: Adjusted Mutual Info Score ####################
    ###########################################################################

    adjusted_mutual_info_score = sklearn.metrics \
        .adjusted_mutual_info_score(labels_true=labels_true,
                                    labels_pred=labels_pred)
    mlops.set_stat("User Defined: Adjusted Mutual Info Score", adjusted_mutual_info_score)

    #########################################################################
    #################### End: Adjusted Mutual Info Score ####################
    #########################################################################

    ####################################################################
    #################### Start: Adjusted Rand Score ####################
    ####################################################################

    adjusted_rand_score = sklearn.metrics \
        .adjusted_rand_score(labels_true=labels_true,
                             labels_pred=labels_pred)
    mlops.set_stat("User Defined: Adjusted Rand Score", adjusted_rand_score)

    ##################################################################
    #################### End: Adjusted Rand Score ####################
    ##################################################################

    #######################################################################
    #################### Start: Calinski Harabaz Score ####################
    #######################################################################

    calinski_harabaz_score = sklearn.metrics \
        .calinski_harabaz_score(X=features, labels=labels_pred)
    mlops.set_stat("User Defined: Calinski Harabaz Score", calinski_harabaz_score)

    #####################################################################
    #################### End: Calinski Harabaz Score ####################
    #####################################################################

    ###################################################################
    #################### Start: Completeness Score ####################
    ###################################################################

    completeness_score = sklearn.metrics \
        .completeness_score(labels_true=labels_true, labels_pred=labels_pred)
    mlops.set_stat("User Defined: Completeness Score", completeness_score)

    #################################################################
    #################### End: Completeness Score ####################
    #################################################################

    ###################################################################
    #################### Start: Contingency Matrix ####################
    ###################################################################

    contingency_matrix = sklearn.metrics.cluster \
        .contingency_matrix(labels_true, labels_pred, eps=None)

    from parallelm.mlops.stats.table import Table

    # list of sorted labels. i.e. [0, 1, 2, ..]
    cm_cols_ordered = sorted(set(labels_pred))
    cm_rows_ordered = sorted(set(labels_true))

    cm_cols_ordered_string = [str(i) for i in cm_cols_ordered]
    cm_rows_ordered_string = [str(i) for i in cm_rows_ordered]

    cm_matrix = Table().name("User Defined: Contingency Matrix").cols(cm_cols_ordered_string)

    for index in range(len(contingency_matrix)):
        cm_matrix.add_row(cm_rows_ordered_string[index], list(contingency_matrix[index]))

    mlops.set_stat(cm_matrix)

    #################################################################
    #################### End: Contingency Matrix ####################
    #################################################################

    ######################################################################
    #################### Start: Fowlkes Mallows Score ####################
    ######################################################################

    fowlkes_mallows_score = sklearn.metrics.fowlkes_mallows_score(labels_true=labels_true, labels_pred=labels_pred,
                                                                  sparse=False)
    mlops.set_stat("User Defined: Fowlkes Mallows Score", fowlkes_mallows_score)

    ####################################################################
    #################### End: Fowlkes Mallows Score ####################
    ####################################################################

    #####################################################################################
    #################### Start: Homogeneity, Completeness, V Measure ####################
    #####################################################################################

    homogeneity, completeness, v_measure = sklearn.metrics \
        .homogeneity_completeness_v_measure(labels_true=labels_true, labels_pred=labels_pred)
    mlops.set_stat("User Defined: Homogeneity", homogeneity)
    mlops.set_stat("User Defined: Completeness", completeness)
    mlops.set_stat("User Defined: V Measure", v_measure)

    ###################################################################################
    #################### End: Homogeneity, Completeness, V Measure ####################
    ###################################################################################

    ##################################################################
    #################### Start: Homogeneity Score ####################
    ##################################################################

    homogeneity_score = sklearn.metrics \
        .homogeneity_score(labels_true=labels_true, labels_pred=labels_pred)
    mlops.set_stat("User Defined: Homogeneity Score", homogeneity_score)

    ################################################################
    #################### End: Homogeneity Score ####################
    ################################################################

    ##################################################################
    #################### Start: Mutual Info Score ####################
    ##################################################################

    mutual_info_score = sklearn.metrics \
        .mutual_info_score(labels_true=labels_true, labels_pred=labels_pred, contingency=None)
    mlops.set_stat("User Defined: Mutual Info Score", mutual_info_score)

    ################################################################
    #################### End: Mutual Info Score ####################
    ################################################################

    #############################################################################
    #################### Start: Normalized Mutual Info Score ####################
    #############################################################################

    normalized_mutual_info_score = sklearn.metrics \
        .normalized_mutual_info_score(labels_true=labels_true,
                                      labels_pred=labels_pred)
    mlops.set_stat("User Defined: Normalized Mutual Info Score", normalized_mutual_info_score)

    ###########################################################################
    #################### End:  Normalized Mutual Info Score ####################
    ###########################################################################

    #################################################################
    #################### Start: Silhouette Score ####################
    #################################################################

    silhouette_score = sklearn.metrics \
        .silhouette_score(X=features, labels=labels_pred, metric="euclidean", sample_size=None, random_state=None)
    mlops.set_stat("User Defined: Silhouette Score", silhouette_score)

    ###############################################################
    #################### End: Silhouette Score ####################
    ###############################################################

    ################################################################
    #################### Start: V Measure Score ####################
    ################################################################

    v_measure_score = sklearn.metrics.v_measure_score(labels_true=labels_true, labels_pred=labels_pred)
    mlops.set_stat("User Defined: V Measure Score", v_measure_score)

    ##############################################################
    #################### End: V Measure Score ####################
    ##############################################################

    # Save the model
    import pickle
    model_file = open(pm_options.output_model, 'wb')
    pickle.dump(kmeans_model, model_file)
    model_file.close()
    # Terminate MLOPs
    mlops.done()


if __name__ == "__main__":
    main()