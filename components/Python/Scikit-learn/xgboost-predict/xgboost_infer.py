from __future__ import print_function

import argparse
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.datasets import make_classification

from parallelm.mlops import StatCategory as st
from parallelm.mlops import mlops as mlops
from parallelm.mlops.predefined_stats import PredefinedStats
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.table import Table


def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", help="# Samples")
    parser.add_argument("--num_features", help="# Features")
    parser.add_argument("--input_file", help="Input Data File")

    parser.add_argument("--ks_threshold", help="KS Threshold")
    parser.add_argument("--psi_threshold", help="PSI Threshold")

    parser.add_argument("--input-model", help="Path of Input Model to Create")
    options = parser.parse_args()
    return options


def get_psi(v1_in, v2_in, num=10):
    """
    calculate PSI.

    :param v1_in: vector 1
    :param v2_in: vector 2
    :param num: number of bins
    :return: PSI Value
    """
    if len(v1_in) < 2:
        v1 = v2_in
        v2 = np.zeros(1)
    elif len(v2_in) == 0:
        v1 = v1_in
        v2 = np.zeros(1)
    else:
        v1 = v1_in
        v2 = v2_in

    rank1 = pd.qcut(v1, num, labels=False) + 1

    basepop1 = pd.DataFrame({'v1': v1, 'rank1': rank1})

    quantiles = basepop1.groupby('rank1').agg({'min', 'max'})
    quantiles['v1'].loc[1][0] = 0

    currpop = pd.DataFrame({'v2': v2, 'rank1': [1] * v2.shape[0]})
    for i in range(2, num + 1):
        currpop['rank1'][currpop['v2'] >= quantiles['v1'].loc[i][0]] = i
        quantiles['v1'].loc[i - 1][1] = quantiles['v1'].loc[i][0]
    quantiles['v1'].loc[num][1] = 1

    basepop2 = basepop1.groupby('rank1').agg({'count'})
    basepop2 = basepop2.rename(columns={'count': 'basenum'})

    currpop2 = currpop.groupby('rank1').agg({'count'})
    currpop2 = currpop2.rename(columns={'count': 'currnum'})

    nbase = basepop1.shape[0]
    ncurr = currpop.shape[0]

    mrged1 = basepop2['v1'].join(currpop2['v2'], how='left')
    mrged1.currnum[mrged1.currnum.isna()] = 0

    mrged2 = mrged1.join(quantiles['v1'], how='left')

    mrged3 = mrged2
    mrged3['basepct'] = mrged3.basenum / nbase
    mrged3['currpct'] = mrged3.currnum / ncurr

    mrged4 = mrged3
    mrged4['psi'] = (mrged4.currpct - mrged4.basepct) * np.log((mrged4.currpct / mrged4.basepct))

    print("Merged DF: {}".format(mrged4))

    tot_PSI = sum(mrged4.psi[mrged4.psi != float('inf')])
    final_table = mrged4
    return tot_PSI, final_table


def main():
    pm_options = parse_args()
    print("PM: Configuration:")
    print("PM: # Sample:                    [{}]".format(pm_options.num_samples))
    print("PM: # Features:                  [{}]".format(pm_options.num_features))

    print("PM: # KS Threshold:              [{}]".format(pm_options.ks_threshold))
    print("PM: # PSI Threshold:             [{}]".format(pm_options.psi_threshold))

    print("PM: # Input File:                [{}]".format(pm_options.input_file))
    print("PM: # Model File:                [{}]".format(pm_options.input_model))

    # Initialize MLOps Library
    mlops.init()
    # Load the model
    if pm_options.input_model is not None:
        try:
            filename = pm_options.input_model
            model_file_obj = open(filename, 'rb')
            mlops.set_stat("# Model Files Used", 1)
        except Exception as e:
            print("Model Not Found")
            print("Got Exception: {}".format(e))
            mlops.set_stat("# Model Files Used", 0)
            mlops.done()
            return 0

    final_model = pickle.load(model_file_obj)

    try:
        data_filename = pm_options.input_file
        data_file_obj = open(data_filename, 'rb')
        data = np.loadtxt(data_file_obj)

        X = data  # select columns 1 through end

    except Exception as e:
        print("Generating Synthetic Data Because {}".format(e))

        # Create synthetic data (Gaussian Distribution, Poisson Distribution and Beta Distribution)
        num_samples = int(pm_options.num_samples)
        num_features = int(pm_options.num_features)

        # Create synthetic data using scikit learn
        X, y = make_classification(n_samples=num_samples,
                                   n_features=num_features,
                                   #                                binary classification only!
                                   n_classes=2,
                                   random_state=42)

        # Add random noise to the data randomly
        import random
        if random.randint(1, 21) / 2 == 0:
            print("Adding Random Noise!")

            noisy_features = np.random.uniform(0, 1) * \
                             np.random.normal(0, 1,
                                              (num_samples, num_features))
            X = X + noisy_features

    # Separate into features and labels
    features = X

    max_ks_requirement = float(pm_options.ks_threshold)
    min_psi_requirement = float(pm_options.psi_threshold)

    # Output Health Statistics to MCenter
    # MLOps API to report the distribution statistics of each feature in the data and compare it automatically with the ones
    mlops.set_data_distribution_stat(features)

    # Output the number of samples being processed using MCenter
    mlops.set_stat(PredefinedStats.PREDICTIONS_COUNT, len(features), st.TIME_SERIES)

    # Accuracy for the chosen model
    pred_labels = final_model.predict(features)
    pred_probs = final_model.predict_proba(features)

    print("Pred Labels: ", pred_labels)  # Remove printout can be huge
    print("Pred Probabilities: ", pred_probs)  # Remove printout can be huge

    # Pred Label distribution
    pred_value, pred_counts = np.unique(pred_labels, return_counts=True)
    pred_label_distribution = np.asarray((pred_value, pred_counts)).T
    # pred_column_names = pred_value.astype(str).tolist()
    print("Pred Label distributions: \n {0}".format(pred_label_distribution))

    # Output Pred label distribution as a BarGraph using MCenter
    pred_bar = BarGraph().name("Pred Label Distribution").cols(
        (pred_label_distribution[:, 0]).astype(str).tolist()).data(
        (pred_label_distribution[:, 1]).tolist())
    mlops.set_stat(pred_bar)

    # Pred Label confidence per label
    label_number = len(pred_counts)
    average_confidence = np.zeros(label_number)
    max_pred_probs = pred_probs.max(axis=1)
    for i in range(0, label_number):
        index_class = np.where(pred_labels == i)[0]
        print(" np.sum(confidence[index_class])", np.sum(max_pred_probs[index_class]))
        print("counts_elements[i] ", pred_counts[i])
        if pred_counts[i] > 0:
            average_confidence[i] = np.sum(max_pred_probs[index_class]) / (float(pred_counts[i]))
        else:
            average_confidence[i] = 0

    # BarGraph showing confidence per class
    pred_values1 = [str(i) for i in pred_value]
    bar = BarGraph().name("Average Confidence Per Class").cols(pred_values1).data(average_confidence.tolist())
    mlops.set_stat(bar)

    # KS for the chosen model
    ks = ks_2samp(max_pred_probs[pred_labels == 1], max_pred_probs[pred_labels == 0])
    ks_stat = ks.statistic
    ks_pvalue = ks.pvalue

    print("KS values: \n Statistics: {} \n pValue: {}\n".format(ks_stat, ks_pvalue))

    # Output KS Stat of the chosen model using MCenter
    if not np.isnan(ks_stat):
        print("printing KS_stat ")
        mlops.set_stat("KS Stat", ks_stat, st.TIME_SERIES)
    else:
        print("not printing KS_stat ")

    # Raising alert if ks-stat goes above required threshold
    if ks_stat >= max_ks_requirement:
        mlops.health_alert("[Inference] KS Violation From Inference Node",
                           "KS Stat Went Above {}. Current KS Stat Is {}".format(max_ks_requirement, ks_stat))

    ks_table = Table().name("KS Stats").cols(["Statistic", "pValue"])
    ks_table.add_row([ks_stat, ks_pvalue])
    mlops.set_stat(ks_table)

    # Calculating PSI
    total_psi, psi_table = get_psi(max_pred_probs[pred_labels == 1], max_pred_probs[pred_labels == 0])

    psi_table_stat = Table().name("PSI Stats").cols(
        ["Base Pop", "Curr Pop",
         "Lower Bound", "Upper Bound", "Base Percent", "Curr Percent",
         "Segment PSI"])

    row_num = 1
    for each_value in psi_table.values:
        str_values = [str(i) for i in each_value]
        psi_table_stat.add_row(str(row_num), str_values)
        row_num += 1

    mlops.set_stat(psi_table_stat)

    print("Total PSI values: \n {}".format(total_psi))

    #     Output Total PSI of the chosen model using MCenter
    mlops.set_stat("Total PSI ", total_psi, st.TIME_SERIES)

    # Raising alert if total_psi goes below required threshold
    if total_psi <= min_psi_requirement:
        mlops.health_alert("[Inference] PSI Violation From Inference Node",
                           "PSI Went Below {}. Current PSI Is {}".format(min_psi_requirement, total_psi))

    # Terminate MLOPs
    mlops.done()


if __name__ == "__main__":
    main()
