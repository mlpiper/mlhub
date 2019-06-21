import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

#from parallelm.mlops import StatCategory as st
#from parallelm.mlops import mlops as mlops
#from parallelm.mlops.stats.bar_graph import BarGraph
#from parallelm.mlops.stats.graph import MultiGraph
#from parallelm.mlops.stats.table import Table


def parse_args():
    """
    Parse arguments from component
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_features", help="# Features", default=21)
    parser.add_argument("--num_samples", help="# Samples", default=1500)
    parser.add_argument("--input_file", help="Input Data File")
    parser.add_argument("--validation_split", help="# Validation Split", default=0.33)

    parser.add_argument("--n_estimators", help="Number of Estimators", default=500)
    parser.add_argument("--max_depth", help="Max Depth", default=7)
    parser.add_argument("--learning_rate", help="Learning Rate", default=0.1)
    parser.add_argument("--min_child_weight", help="Min Child Weight", default=1)
    parser.add_argument("--objective", help="Objective", default="binary:logistic")
    parser.add_argument("--gamma", help="Gamma", default=0)
    parser.add_argument("--max_delta_step", help="Max Delta Step", default=0)
    parser.add_argument("--subsample", help="Subsample", default=1)
    parser.add_argument("--reg_alpha", help="Reg Alpha", default=0)
    parser.add_argument("--reg_lambda", help="Reg Lambda", default=0)
    parser.add_argument("--scale_pos_weight", help="Scale Pos Weight", default=1)
    parser.add_argument("--auc_threshold", help="AUC Threshold", default=1)
    parser.add_argument("--ks_threshold", help="KS Threshold", default=1)
    parser.add_argument("--psi_threshold", help="PSI Threshold", default=1)

    parser.add_argument("--output-model", help="Data File to Save Model", default="/tmp/f")
    options = parser.parse_args()
    return options


def get_psi(v1_in, v2_in, num=10):
    """
    calculate PSI.

    :param v1: vector 1
    :param v2: vector 2
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

    print("PM: # Validation Split:          [{}]".format(pm_options.validation_split))

    print("PM: # AUC Threshold:             [{}]".format(pm_options.auc_threshold))
    print("PM: # KS Threshold:              [{}]".format(pm_options.ks_threshold))
    print("PM: # PSI Threshold:             [{}]".format(pm_options.psi_threshold))

    print("PM: # Estimators:                [{}]".format(pm_options.n_estimators))
    print("PM: # Max Depth:                 [{}]".format(pm_options.max_depth))
    print("PM: # Learning Rate:             [{}]".format(pm_options.learning_rate))
    print("PM: # Min Child Weight:          [{}]".format(pm_options.min_child_weight))
    print("PM: # Objective:                 [{}]".format(pm_options.objective))
    print("PM: # Gamma:                     [{}]".format(pm_options.gamma))
    print("PM: # Max Delta Step:            [{}]".format(pm_options.max_delta_step))
    print("PM: # Subsample:                 [{}]".format(pm_options.subsample))
    print("PM: # Reg Alpha:                 [{}]".format(pm_options.reg_alpha))
    print("PM: # Reg Lambda:                [{}]".format(pm_options.reg_lambda))
    print("PM: # Scale Pos Weight:          [{}]".format(pm_options.scale_pos_weight))

    print("PM: # Input File:                [{}]".format(pm_options.input_file))
    print("PM: Output model:                [{}]".format(pm_options.output_model))

    min_auc_requirement = float(pm_options.auc_threshold)
    max_ks_requirement = float(pm_options.ks_threshold)
    min_psi_requirement = float(pm_options.psi_threshold)

    # Initialize MLOps Library
    #mlops.init()

    try:
        data_filename = pm_options.input_file
        data_file_obj = open(data_filename, 'rb')
        data = np.loadtxt(data_file_obj)

        X = data[:, 1:]  # select columns 1 through end
        y = data[:, 0]

    except Exception as e:
        print("Generating Synthetic Data Because {}".format(e))

        # Create synthetic data (Gaussian Distribution, Poisson Distribution and Beta Distribution)
        num_samples = int(pm_options.num_samples)
        num_features = int(pm_options.num_features)

        # Create synthetic data using scikit learn
        X, y = make_classification(n_samples=num_samples,
                                   n_features=num_features,
                                   # binary classification only!
                                   n_classes=2,
                                   random_state=42)

        print("Adding Random Noise!")

        noisy_features = np.random.uniform(0, 1) * \
                         np.random.normal(0, 1,
                                          (num_samples, num_features))
        X = X + noisy_features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(pm_options.validation_split),
                                                        random_state=42)

    import xgboost as xgb

    # Create a model that should be deployed into production
    final_model = xgb.XGBClassifier(max_depth=int(pm_options.max_depth),
                                    min_child_weight=int(pm_options.min_child_weight),
                                    learning_rate=float(pm_options.learning_rate),
                                    n_estimators=int(pm_options.n_estimators),
                                    silent=True,
                                    objective=str(pm_options.objective),
                                    gamma=float(pm_options.gamma),
                                    max_delta_step=int(pm_options.max_delta_step),
                                    subsample=float(pm_options.subsample),
                                    colsample_bytree=1,
                                    colsample_bylevel=1,
                                    reg_alpha=float(pm_options.reg_alpha),
                                    reg_lambda=float(pm_options.reg_lambda),
                                    scale_pos_weight=float(pm_options.scale_pos_weight),
                                    seed=1,
                                    missing=None)

    final_model.fit(X_train, y_train)

    # Output Health Statistics to MCenter
    # MLOps API to report the distribution statistics of each feature in the data
    #mlops.set_data_distribution_stat(X_train)

    # Accuracy for the chosen model
    pred_labels = final_model.predict(X_test)
    pred_probs = final_model.predict_proba(X_test)

    print("Pred Labels: ", pred_labels)
    print("Pred Probabilities: ", pred_probs)

    accuracy = accuracy_score(y_test, pred_labels)
    print("Accuracy values: \n {0}".format(accuracy))
    # Output accuracy of the chosen model using MCenter
    #mlops.set_stat("Accuracy", accuracy, st.TIME_SERIES)

    # Label distribution in training
    value, counts = np.unique(y_test, return_counts=True)
    label_distribution = np.asarray((value, counts)).T
    # column_names = value.astype(str).tolist()
    print("Validation Actual Label distributions: \n {0}".format(label_distribution))

    # Output Label distribution as a BarGraph using MCenter
    #bar = BarGraph().name("Validation Actual Label Distribution").cols(
    #    (label_distribution[:, 0]).astype(str).tolist()).data(
    #    (label_distribution[:, 1]).tolist())
    #mlops.set_stat(bar)

    # Pred Label distribution in training
    pred_value, pred_counts = np.unique(pred_labels, return_counts=True)
    pred_label_distribution = np.asarray((pred_value, pred_counts)).T
    # pred_column_names = pred_value.astype(str).tolist()
    print("Validation Prediction Label Distributions: \n {0}".format(pred_label_distribution))

    # Output Pred label distribution as a BarGraph using MCenter
    #pred_bar = BarGraph().name("Validation Prediction Label Distributions").cols(
    #    (pred_label_distribution[:, 0]).astype(str).tolist()).data(
    #    (pred_label_distribution[:, 1]).tolist())
    #mlops.set_stat(pred_bar)

    # ROC for the chosen model
    roc_auc = roc_auc_score(y_test, pred_probs[:, 1])
    print("ROC AUC values: \n {}".format(roc_auc))

    #     Output ROC of the chosen model using MCenter
    #mlops.set_stat("ROC AUC", roc_auc, st.TIME_SERIES)

    #if roc_auc <= min_auc_requirement:
        #mlops.health_alert("[Training] AUC Violation From Training Node",
        #                   "AUC Went Below {}. Current AUC Is {}".format(min_auc_requirement, roc_auc))

    # ROC Curve
    fpr, tpr, thr = roc_curve(y_test, pred_probs[:, 1])
    #cg = MultiGraph().name("Receiver Operating Characteristic ").set_continuous()
    #cg.add_series(label='Random Curve ''', x=fpr.tolist(), y=fpr.tolist())
    #cg.add_series(label='ROC Curve (Area = {0:0.2f})'''.format(roc_auc), x=fpr.tolist(), y=tpr.tolist())
    #cg.x_title('False Positive Rate')
    #cg.y_title('True Positive Rate')
    #mlops.set_stat(cg)

    max_pred_probs = pred_probs.max(axis=1)

    # KS for the chosen model
    ks = ks_2samp(max_pred_probs[y_test == 1], max_pred_probs[y_test == 0])
    ks_stat = ks.statistic
    ks_pvalue = ks.pvalue

    print("KS values: \n Statistics: {} \n pValue: {}\n".format(ks_stat, ks_pvalue))

    # Output KS Stat of the chosen model using MCenter
    #mlops.set_stat("KS Stat", ks_stat, st.TIME_SERIES)

    # Raising alert if ks-stat goes above required threshold
    #if ks_stat >= max_ks_requirement:
    #    mlops.health_alert("[Training] KS Violation From Training Node",
    #                       "KS Stat Went Above {}. Current KS Stat Is {}".format(max_ks_requirement, ks_stat))

    #ks_table = Table().name("KS Stats").cols(["Statistic", "pValue"])
    #ks_table.add_row([ks_stat, ks_pvalue])
    #mlops.set_stat(ks_table)

    # Calculating PSI
    total_psi, psi_table = get_psi(max_pred_probs[y_test == 1], max_pred_probs[y_test == 0])

    #psi_table_stat = Table().name("PSI Stats").cols(
    #    ["Base Pop", "Curr Pop", "Lower Bound", "Upper Bound", "Base Percent", "Curr Percent",
    #     "Segment PSI"])

    row_num = 1
    #for each_value in psi_table.values:
    #    str_values = [str(i) for i in each_value]
    #    psi_table_stat.add_row(str(row_num), str_values)
    #    row_num += 1

    #mlops.set_stat(psi_table_stat)

    print("Total PSI values: \n {}".format(total_psi))

    # Output Total PSI of the chosen model using MCenter
    #mlops.set_stat("Total PSI ", total_psi, st.TIME_SERIES)

    # Raising alert if total_psi goes below required threshold
    #if total_psi <= min_psi_requirement:
    #    mlops.health_alert("[Training] PSI Violation From Training Node",
    #                       "PSI Went Below {}. Current PSI Is {}".format(min_psi_requirement, total_psi))

    # Save the model
    import pickle
    model_file = open(pm_options.output_model, 'wb')
    pickle.dump(final_model, model_file)
    model_file.close()
    # Terminate MLOPs
    #mlops.done()


if __name__ == "__main__":
    main()
