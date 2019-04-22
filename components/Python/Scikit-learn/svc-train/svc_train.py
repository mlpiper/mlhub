import argparse

import numpy as np
import sklearn
from parallelm.mlops import mlops as mlops
# use below import if user wants to user ClassificationMetrics predefined metrics names.
from parallelm.mlops.metrics_constants import ClassificationMetrics
from parallelm.mlops.stats.bar_graph import BarGraph
from sklearn.datasets import make_classification
from sklearn.svm import SVC


def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="# samples")
    parser.add_argument("--num_features", help="# features")
    parser.add_argument("--num_classes", help="# classes")

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
    print("PM: # Classes:                   [{}]".format(pm_options.num_classes))

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
    num_classes = int(pm_options.num_classes)

    # Create synthetic data using scikit learn
    X, y = make_classification(n_samples=num_samples,
                               n_features=num_features,
                               n_informative=2,
                               n_redundant=1,
                               n_classes=num_classes,
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
                      probability=True,
                      kernel=pm_options.kernel,
                      degree=int(pm_options.degree),
                      gamma=str(pm_options.gamma),
                      tol=float(pm_options.tol),
                      max_iter=int(pm_options.max_iter))

    final_model.fit(features, labels)

    value, counts = np.unique(labels, return_counts=True)
    label_distribution = np.asarray((value, counts)).T

    # Output actual label distribution as a BarGraph using MCenter
    bar = BarGraph().name("User Defined: Actual Label Distribution") \
        .cols((label_distribution[:, 0]).astype(str).tolist()) \
        .data((label_distribution[:, 1]).tolist())
    mlops.set_stat(bar)

    labels_pred = final_model.predict(features)

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

    labels_ordered = sorted(set(labels))

    ################################################################
    #################### Start: Output Accuracy ####################
    ################################################################

    accuracy = final_model.score(features, labels)

    #################### OLD WAY ####################
    # First Way
    #
    # # Output accuracy of the chosen model using MCenter
    # mlops.set_stat("User Defined: Accuracy", accuracy, st.TIME_SERIES)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.ACCURACY_SCORE, accuracy)

    # OR

    # Third Way
    mlops.metrics.accuracy_score(y_true=labels, y_pred=labels_pred)
    #################### DONE NEW WAY ####################

    ##############################################################
    #################### End: Output Accuracy ####################
    ##############################################################

    ################################################################
    #################### Start: Output AUC ####################
    ################################################################

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, labels_pred, pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)

    #################### OLD WAY ####################
    # First Way
    #
    # # Output auc of the chosen model using MCenter
    # mlops.set_stat("User Defined: AUC", auc)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.AUC, auc)

    # OR

    # Third Way
    mlops.metrics.auc(x=fpr, y=tpr)
    #################### DONE NEW WAY ####################

    ##############################################################
    #################### End: Output AUC ####################
    ##############################################################

    ################################################################
    #################### Start: Output AUC ####################
    ################################################################

    labels_decision_score = final_model.decision_function(features)
    aps = sklearn.metrics.average_precision_score(labels, labels_decision_score)

    #################### OLD WAY ####################
    # First Way
    #
    # # Output aps of the chosen model using MCenter
    # mlops.set_stat("User Defined: Average Precision Score", aps)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.AVERAGE_PRECISION_SCORE, aps)

    # OR

    # Third Way
    mlops.metrics.average_precision_score(y_true=labels, y_score=labels_decision_score)
    #################### DONE NEW WAY ####################

    ##############################################################
    #################### End: Output AUC ####################
    ##############################################################

    #########################################################################
    #################### Start: Output Balanced Accuracy ####################
    #########################################################################

    bas = sklearn.metrics.balanced_accuracy_score(labels, labels_pred)

    #################### OLD WAY ####################
    # First Way
    #
    # # Output bas of the chosen model using MCenter
    # mlops.set_stat("User Defined: Balanced Accuracy Score", bas)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.BALANCED_ACCURACY_SCORE, data=bas)

    # OR

    # Third Way
    mlops.metrics.balanced_accuracy_score(y_true=labels, y_pred=labels_pred)
    #################### DONE NEW WAY ####################

    #######################################################################
    #################### End: Output Balanced Accuracy ####################
    #######################################################################

    ########################################################################
    #################### Start: Output Brier Score Loss ####################
    ########################################################################

    labels_prob = final_model.predict_proba(features)
    label_pos_class_prob = list(map(lambda x: x[1], labels_prob))
    bsl = sklearn.metrics.brier_score_loss(labels, label_pos_class_prob, pos_label=1)

    #################### OLD WAY ####################
    # First Way
    #
    # # Output bsl of the chosen model using MCenter
    # mlops.set_stat("User Defined: Brier Score Loss", bsl)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.BRIER_SCORE_LOSS, data=bsl)

    # OR

    # Third Way
    mlops.metrics.brier_score_loss(y_true=labels, y_prob=label_pos_class_prob, pos_label=1)
    #################### DONE NEW WAY ####################

    ######################################################################
    #################### End: Output Brier Score Loss ####################
    ######################################################################

    #############################################################################
    #################### Start: Output Classification Report ####################
    #############################################################################
    cr = sklearn.metrics.classification_report(labels, labels_pred)
    print("Classification Report\n{}".format(cr))
    #################### OLD WAY ####################
    # First Way
    #
    # from parallelm.mlops.stats.table import Table
    #
    # arrayReport = list()
    # for row in cr.split("\n"):
    #     parsed_row = [x for x in row.split("  ") if len(x) > 0]
    #     if len(parsed_row) > 0:
    #         arrayReport.append(parsed_row)
    #
    # header = arrayReport[0]
    # cr_table = Table().name("User Defined: Classification Report").cols(header)
    #
    # for index in range(1, len(arrayReport)):
    #     row_title = arrayReport[index][0]
    #     row_value = arrayReport[index][:-1]
    #     cr_table.add_row(row_title, row_value)
    #
    # # output classification report using MCenter
    # mlops.set_stat(cr_table)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.CLASSIFICATION_REPORT, data=cr)

    # OR

    # Third Way
    mlops.metrics.classification_report(labels, labels_pred)
    #################### DONE NEW WAY ####################

    ###########################################################################
    #################### End: Output Classification Report ####################
    ###########################################################################

    #########################################################################
    #################### Start: Output Cohen Kappa Score ####################
    #########################################################################

    cks = sklearn.metrics.cohen_kappa_score(labels, labels_pred)

    #################### OLD WAY ####################
    # First Way
    #
    # # Output cks of the chosen model using MCenter
    # mlops.set_stat("User Defined: Cohen Kappa Score", cks)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.COHEN_KAPPA_SCORE, data=cks)

    # OR

    # Third Way
    mlops.metrics.cohen_kappa_score(labels, labels_pred)
    #################### DONE NEW WAY ####################

    #######################################################################
    #################### End: Output Cohen Kappa Score ####################
    #######################################################################

    ########################################################################
    #################### Start: Output Confusion Matrix ####################
    ########################################################################

    cm = sklearn.metrics.confusion_matrix(labels, labels_pred, labels=labels_ordered)

    #################### OLD WAY ####################
    # First Way
    # from parallelm.mlops.stats.table import Table

    # labels_string = [str(i) for i in labels_ordered]
    # cm_matrix = Table().name("User Defined: Confusion Matrix").cols(labels_string)
    #
    # for index in range(len(cm)):
    #     cm_matrix.add_row(labels_string[index], list(cm[index]))
    #
    # mlops.set_stat(cm_matrix)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.CONFUSION_MATRIX, cm, labels=labels_ordered)

    # OR

    # Third Way
    mlops.metrics.confusion_matrix(y_true=labels, y_pred=labels_pred, labels=labels_ordered)
    #################### DONE NEW WAY ####################

    ######################################################################
    #################### End: Output Confusion Matrix ####################
    ######################################################################

    ################################################################
    #################### Start: Output F1 Score ####################
    ################################################################

    f1 = sklearn.metrics.f1_score(labels, labels_pred)

    #################### OLD WAY ####################
    # First Way
    #
    # # Output f1 score of the chosen model using MCenter
    # mlops.set_stat("User Defined: F1 Score", f1)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.F1_SCORE, data=f1)

    # OR

    # Third Way
    mlops.metrics.f1_score(labels, labels_pred, pos_label=1)
    #################### DONE NEW WAY ####################

    ##############################################################
    #################### End: Output F1 Score ####################
    ##############################################################

    ################################################################
    #################### Start: Output FBeta Score ####################
    ################################################################

    fbeta = sklearn.metrics.fbeta_score(labels, labels_pred, beta=0.5)

    #################### OLD WAY ####################
    # First Way
    #
    # # Output fbeta score of the chosen model using MCenter
    # mlops.set_stat("User Defined: F-beta Score", fbeta)
    #################### DONE OLD WAY ####################

    #################### NEW WAY ####################
    # Second Way
    mlops.set_stat(ClassificationMetrics.FBETA_SCORE, data=fbeta)

    # OR

    # Third Way
    mlops.metrics.fbeta_score(labels, labels_pred, pos_label=1, beta=0.5)
    #################### DONE NEW WAY ####################

    #################################################################
    #################### End: Output FBeta Score ####################
    #################################################################

    # Save the model
    import pickle
    model_file = open(pm_options.output_model, 'wb')
    pickle.dump(final_model, model_file)
    model_file.close()
    # Terminate MLOPs
    mlops.done()


if __name__ == "__main__":
    main()
