import numpy as np
import pandas as pd
import argparse
import pickle
from parallelm.mlops import StatCategory as st
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.table import Table
from scipy.stats import ks_2samp


def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--auc_threshold", help="AUC Threshold")
    parser.add_argument("--ks_threshold", help="KS Threshold")
    parser.add_argument("--psi_threshold", help="PSI Threshold")
    parser.add_argument("--input_file", help="Input Data File")

    parser.add_argument("--input-model", help="Path of Input Model to Create")
    options = parser.parse_args()
    return options


def export_feature_importance(final_model, column_names, num_features, title_name):
    """
    This function provides a feature importance at MCenter data scientist view
    :param final_model: Pipeline model (Assume - Feature_Eng + Algo)
    :param column_names: Column names of the input dataframe.
    :param num_features: Number of fefatures to shpw.
    :param title_name: Title of the bar Graph
    :return:
    """
    model_oh = final_model.steps[0][1].features
    trans_feature_names = []
    for mod_el in range(0,len(model_oh)):
        if("OneHotEncoder" in model_oh[mod_el][1].__class__.__name__):
            trans_feature_names += list(model_oh[mod_el][1].get_feature_names([column_names[mod_el]]))
        else:
            trans_feature_names.append(column_names[mod_el])
    trans_feature_names1 = np.asarray(trans_feature_names)
    model_FE_index = np.argsort(final_model.steps[-1][1].feature_importances_)[::-1][:num_features]
    feat_eng = pd.DataFrame({'Name': trans_feature_names1[model_FE_index],
                             'Importance': final_model.steps[-1][1].feature_importances_[model_FE_index]})
    print("Feature Importance for " + str(title_name) + "\n", feat_eng)
    # Output Feature Importance as a BarGraph using MCenter
    export_bar_table(trans_feature_names1[model_FE_index],
                     final_model.steps[-1][1].feature_importances_[model_FE_index],
                     "Feature Importance for " + str(title_name))


def get_psi(v1, v2, num1=10):
    """
    calculate PSI.

    :param v1: vector 1
    :param v2: vector 2
    :param num1: number of bins
    :return: PSI Value
    """
    rank1 = pd.qcut(v1, num1, labels=False, duplicates="drop") + 1
    num = min(num1, max(rank1))

    basepop1 = pd.DataFrame({'v1': v1, 'rank1': rank1})

    quantiles = basepop1.groupby('rank1').agg({'min', 'max'})
    quantiles.loc[1, 'v1'][0] = 0

    currpop = pd.DataFrame({'v2': v2, 'rank1': [1] * v2.shape[0]})
    for i in range(2, num + 1):
        currpop.loc[currpop['v2'] >= quantiles['v1'].loc[i][0], 'rank1'] = i
        quantiles.loc[i - 1, 'v1'][1] = quantiles.loc[i, 'v1'][0]
    quantiles.loc[num, 'v1'][1] = 1

    basepop2 = basepop1.groupby('rank1').agg({'count'})
    basepop2 = basepop2.rename(columns={'count': 'basenum'})

    currpop2 = currpop.groupby('rank1').agg({'count'})
    currpop2 = currpop2.rename(columns={'count': 'currnum'})

    nbase = basepop1.shape[0]
    ncurr = currpop.shape[0]

    mrged1 = basepop2['v1'].join(currpop2['v2'], how='left')
    if mrged1.shape[0] > 1:
        mrged1.loc[mrged1.currnum.isna(), "currnum"] = 0

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


def export_bar_table(bar_names, bar_data, title_name):
    """
    This function provides a bar_graph for a bar type data at MCenter data scientist view
    :param bar_names: Bar graph names
    :param bar_data: Bar graph data.
    :param title_name: Title of the bar Graph
    :return:
    """
    bar_graph_data = BarGraph().name(title_name).cols(
        bar_names.astype(str).tolist()).data(
        bar_data.tolist())
    mlops.set_stat(bar_graph_data)


def main():
    pm_options = parse_args()
    print("PM: Configuration:")

    print("PM: # KS Threshold:              [{}]".format(pm_options.ks_threshold))
    print("PM: # PSI Threshold:             [{}]".format(pm_options.psi_threshold))

    print("PM: # Input File:                [{}]".format(pm_options.input_file))
    print("PM: # Model File:                [{}]".format(pm_options.input_model))

    max_ks_requirement = float(pm_options.ks_threshold)
    min_psi_requirement = float(pm_options.psi_threshold)

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

    # Loading the data
    loan_df = pd.read_csv(pm_options.input_file)
    X = loan_df

    # Cleaning NAs
    mlops.set_data_distribution_stat(loan_df)
    print("dataset_size = ", loan_df.shape[0])
    print("number of NAs per columns = \n",  loan_df.isnull().sum())
    loan_df = loan_df.dropna()
    print("dataset_size without NA rows= ", loan_df.shape[0])

    # ## Inference
    pred_labels = final_model.predict(X)
    pred_probs = final_model.predict_proba(X)

    # Prediction distribution and prediction confidence distribution
    pred_value, pred_counts = np.unique(pred_labels, return_counts=True)
    pred_label_distribution = np.asarray((pred_value, pred_counts)).T
    print("XGBoost Inference Prediction Label Distributions: \n {0}".format(pred_label_distribution))
    export_bar_table(pred_label_distribution[:, 0],
                     pred_label_distribution[:, 1],
                     "Inference - XGBoost Prediction Distribution")

    # Pred confidence per label
    label_number = len(pred_counts)
    average_confidence = np.zeros(label_number)
    max_pred_probs = pred_probs.max(axis=1)
    for i in range(0, label_number):
        index_class = np.where(pred_labels == i)[0]
        if pred_counts[i] > 0:
            average_confidence[i] = np.sum(max_pred_probs[index_class])/(float(pred_counts[i]))
        else:
            average_confidence[i] = 0
    print("XGBoost Validation Average Prediction confidence per label: \n {0}".format(average_confidence))
    # Output Pred label distribution as a BarGraph using MCenter
    export_bar_table(pred_value, average_confidence,
                     "Validation - XGBoost Average confidence per class")

    # Feature importance comparison
    export_feature_importance(final_model, list(X.columns), 5, "XGBoost")

    # KS Analysis
    max_pred_probs = pred_probs.max(axis=1)
    y_test0 = np.where(pred_labels == 0)[0]
    y_test1 = np.where(pred_labels == 1)[0]
    ks = ks_2samp(max_pred_probs[y_test0], max_pred_probs[y_test1])
    ks_stat = ks.statistic
    ks_pvalue = ks.pvalue
    print("KS values for XGBoost: \n Statistics: {} \n pValue: {}\n".format(ks_stat, ks_pvalue))
    # Output KS Stat of the chosen model using MCenter
    mlops.set_stat("KS Stats for XGBoost", ks_stat, st.TIME_SERIES)
    # raising alert if ks-stat goes above required threshold
    if ks_stat >= max_ks_requirement:
        mlops.health_alert("[Training] KS Violation From Training Node",
                           "KS Stat Went Above {}. Current KS Stat Is {}".format(max_ks_requirement,
                                                                                 ks_stat))
    ks_table = Table().name("KS Stats").cols(["Statistic", "pValue"])
    ks_table.add_row([ks_stat, ks_pvalue])
    mlops.set_stat(ks_table)

    # PSI Analysis
    total_psi, psi_table = get_psi(max_pred_probs[y_test0], max_pred_probs[y_test1])
    psi_table_stat = Table().name("PSI Stats").cols(
        ["Base Pop", "Curr Pop", "Lower Bound", "Upper Bound", "Base Percent", "Curr Percent",
         "Segment PSI"])
    row_num = 1
    for each_value in psi_table.values:
        str_values = [str(i) for i in each_value]
        psi_table_stat.add_row(str(row_num), str_values)
        row_num += 1
    mlops.set_stat(psi_table_stat)
    print("Total XGBoost PSI values: \n {}".format(total_psi))
    print("XGBoost PSI Stats: \n {}".format(psi_table))
    #     Output Total PSI of the chosen model using MCenter
    mlops.set_stat("Total PSI ", total_psi, st.TIME_SERIES)

    if total_psi >= min_psi_requirement:
        mlops.health_alert("[Training] PSI Violation From Training Node",
                           "PSI Went Below {}. Current PSI Is {}".format(min_psi_requirement,
                                                                         total_psi))

    # ## Finish the program
    mlops.done()


if __name__ == "__main__":
    main()
