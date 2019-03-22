import numpy as np
import pandas as pd
import argparse

from parallelm.mlops import StatCategory as st
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.table import Table
from parallelm.mlops.stats.graph import MultiGraph
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def parse_args():
    """
    Parse arguments from component
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_split", help="# Validation Split")

    parser.add_argument("--n_estimators", help="Number of Estimators")
    parser.add_argument("--max_depth", help="Max Depth")
    parser.add_argument("--learning_rate", help="Learning Rate")
    parser.add_argument("--min_child_weight", help="Min Child Weight")
    parser.add_argument("--objective", help="Objective")
    parser.add_argument("--gamma", help="Gamma")
    parser.add_argument("--max_delta_step", help="Max Delta Step")
    parser.add_argument("--subsample", help="Subsample")
    parser.add_argument("--reg_alpha", help="Reg Alpha")
    parser.add_argument("--reg_lambda", help="Reg Lambda")
    parser.add_argument("--scale_pos_weight", help="Scale Pos Weight")
    parser.add_argument("--auc_threshold", help="AUC Threshold")
    parser.add_argument("--ks_threshold", help="KS Threshold")
    parser.add_argument("--psi_threshold", help="PSI Threshold")

    parser.add_argument("--input_file", help="Input Data File")

    parser.add_argument("--output-model", help="Data File to Save Model")
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


def export_confusion_table(confmat, algo):
    """
    This function provides the confusion matrix as a table in at MCenter data scientist view
    :param confmat: Confusion matrix
    :param algo: text for the algorithm type
    :return:
    """

    tbl = Table()\
        .name("Confusion Matrix for " + str(algo))\
        .cols(["Predicted label: " + str(i) for i in range(0, confmat.shape[0])])
    for i in range(confmat.shape[1]):
        tbl.add_row("True Label: " + str(i), [str(confmat[i, j]) for j in range(0, confmat.shape[0])])
    mlops.set_stat(tbl)


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


def export_classification_report(class_rep, algo):
    """
    This function provides the classification report as a table in at MCenter data scientist view
    :param class_rep: Classification report data
    :param algo: text for the algorithm type
    :return:
    """
    col_keys = []
    row_keys = []
    class_tlb = []
    add_col_keys = True
    for row_key in class_rep.keys():
        row_keys.append(str(row_key))
        class_tlb_row = []
        class_row = class_rep[row_key]
        for col_key in class_row.keys():
            if add_col_keys:
                col_keys.append(str(col_key))
            class_tlb_row.append(str(class_row[col_key]))
        add_col_keys = False
        class_tlb.append(class_tlb_row)

    tbl = Table().name("Classification Report "+str(algo)).cols(col_keys)
    for i in range(len(row_keys)):
        tbl.add_row(row_keys[i], class_tlb[i])
    mlops.set_stat(tbl)


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


def main():
    pm_options = parse_args()

    print("PM: Configuration:")

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

    # mlops Init
    mlops.init()

    # Loading and cleaning the data
    # This section goes though the various stages of loading and cleaning the data:
    loan_df = pd.read_csv(pm_options.input_file)

    # Cleaning NAs
    print("dataset_size = ", loan_df.shape[0])
    mlops.set_data_distribution_stat(loan_df)
    print("number of NAs per columns = ",  loan_df.isnull().sum())
    loan_df = loan_df.dropna()
    print("dataset_size without NA rows= ", loan_df.shape[0])

    # Marking the label field. remove it from the features set:
    y = loan_df["bad_loan"]
    X = loan_df.drop("bad_loan", axis=1)

    from sklearn_pandas import DataFrameMapper

    # Splitting the data to train and test sets:
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=float(pm_options.validation_split),
                                                        random_state=42)

    All_columns = X_train.columns.tolist()
    categorical_columns = ["verification_status", "addr_state", "purpose", "home_ownership", "term"]
    mapper_list =[]
    for d in All_columns:
        if d in categorical_columns:
            mapper_list.append(([d], OneHotEncoder(handle_unknown='ignore')))
        else:
            mapper_list.append(([d], MinMaxScaler()))

    mapper = DataFrameMapper(mapper_list)

    # ## Training
    # XGBoost Training:
    import xgboost as xgb
    xgboost_model = xgb.XGBClassifier(max_depth=int(pm_options.max_depth),
                                    min_child_weight=int(pm_options.min_child_weight),
                                    learning_rate=float(pm_options.learning_rate),
                                    n_estimators=int(pm_options.n_estimators),
                                    silent=True,
                                    objective=pm_options.objective,
                                    gamma=float(pm_options.gamma),
                                    max_delta_step=int(pm_options.max_delta_step),
                                    subsample=float(pm_options.subsample),
                                    colsample_bytree=1,
                                    colsample_bylevel=1,
                                    reg_alpha=float(pm_options.reg_alpha),
                                    reg_lambda=float(pm_options.reg_lambda),
                                    scale_pos_weight=float(pm_options.scale_pos_weight),
                                    seed=1,
                                    n_jobs=1,
                                    missing=None)

    final_model = Pipeline([("mapper", mapper), ("xgboost", xgboost_model)])

    final_model.fit(X_train, y_train)
    # Random Forest Training
    from sklearn.ensemble import RandomForestClassifier
    rf_only_model = RandomForestClassifier(n_estimators=int(pm_options.n_estimators), max_depth=int(pm_options.max_depth)+3, random_state=42, n_jobs=1, class_weight="balanced")
    rf_model = Pipeline([("mapper", mapper), ("rf", rf_only_model)])

    rf_model.fit(X_train, y_train)

    # ## Statistics on Test Dataset

    # Prediction and prediction distribution
    pred_labels = final_model.predict(X_test)
    pred_probs = final_model.predict_proba(X_test)
    rf_pred_labels = rf_model.predict(X_test)
    rf_pred_probs = rf_model.predict_proba(X_test)

    # Accuracy calculation
    # Accuracy for the xgboost model
    accuracy = accuracy_score(y_test, pred_labels)
    print("XGBoost Accuracy value: {0}".format(accuracy))
    #     Output accuracy of the chosen model using MCenter
    mlops.set_stat("XGBoost Accuracy", accuracy, st.TIME_SERIES)

    # Accuracy for the RF model
    rf_accuracy = accuracy_score(y_test, rf_pred_labels)
    print("RF Accuracy value: {0}".format(rf_accuracy))
    #     Output accuracy of the chosen model using MCenter
    mlops.set_stat("RF Accuracy", rf_accuracy, st.TIME_SERIES)

    # Label distribution:
    # Label distribution in training
    value, counts = np.unique(y_test, return_counts=True)
    label_distribution = np.asarray((value, counts)).T
    print("Validation Actual Label distributions: \n {0}".format(label_distribution))
    # Output Label distribution as a BarGraph using MCenter
    export_bar_table(label_distribution[:,0], label_distribution[:,1], "Validation - Actual Label Distribution")

    # Prediction distribution and prediction confidence distribution
    # Pred Label distribution in training
    pred_value, pred_counts = np.unique(pred_labels, return_counts=True)
    pred_label_distribution = np.asarray((pred_value, pred_counts)).T
    print("XGBoost Validation Prediction Label Distributions: \n {0}".format(pred_label_distribution))
    # Output Pred label distribution as a BarGraph using MCenter
    export_bar_table(pred_label_distribution[:,0], pred_label_distribution[:,1], "Validation - XGBoost Prediction Distribution")

    rf_pred_value, rf_pred_counts = np.unique(rf_pred_labels, return_counts=True)
    rf_pred_label_distribution = np.asarray((rf_pred_value, rf_pred_counts)).T
    # pred_column_names = pred_value.astype(str).tolist()
    print("RF Validation Prediction Label Distributions: \n {0}".format(rf_pred_label_distribution))

    # Output Pred label distribution as a BarGraph using MCenter
    export_bar_table(rf_pred_label_distribution[:,0], rf_pred_label_distribution[:,1], "Validation - RF Prediction Distribution")

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

    #  Pred confidence per label
    rf_label_number = len(rf_pred_counts)
    rf_average_confidence = np.zeros(rf_label_number)
    rf_max_pred_probs = rf_pred_probs.max(axis=1)
    for i in range(0, rf_label_number):
        rf_index_class = np.where(rf_pred_labels == i)[0]
        if rf_pred_counts[i] > 0:
            rf_average_confidence[i] = np.sum(rf_max_pred_probs[rf_index_class])/(float(rf_pred_counts[i]))
        else:
            rf_average_confidence[i] = 0
    print("RF Validation Average Prediction confidence per label: \n {0}".format(rf_average_confidence))

    # Output Pred label distribution as a BarGraph using MCenter
    export_bar_table(pred_value, average_confidence, "Validation - XGBoost Average confidence per class")
    export_bar_table(rf_pred_value, rf_average_confidence, "Validation - RF Average confidence per class")

    # Confusion Matrix
    # XGBoost Confusion Matrix
    confmat = confusion_matrix(y_true=y_test, y_pred=pred_labels)
    print("Confusion Matrix for XGBoost: \n {0}".format(confmat))
    # Output Confusion Matrix as a Table using MCenter
    export_confusion_table(confmat, "XGBoost")
    # RF Confusion Matrix
    rf_confmat = confusion_matrix(y_true=y_test, y_pred=rf_pred_labels)
    print("Confusion Matrix for RF: \n {0}".format(rf_confmat))
    # Output Confusion Matrix as a Table using MCenter
    export_confusion_table(rf_confmat, "RF")

    # Classification Report
    # XGBoost Classification Report
    class_rep = classification_report(y_true=y_test, y_pred=pred_labels, output_dict=True)
    print("XGBoost Classification Report: \n {0}".format(class_rep))
    # RF Classification Report
    rf_class_rep = classification_report(y_true=y_test, y_pred=rf_pred_labels, output_dict=True)
    print("RF Classification Report: \n {0}".format(rf_class_rep))
    # Output Classification Report as a Table using MCenter
    export_classification_report(class_rep, "XGBoost")
    export_classification_report(rf_class_rep, "RF")

    # AUC and ROC Curves
    # ROC for XGBoost model
    roc_auc = roc_auc_score(y_test, pred_probs[:, 1])
    print("XGBoost ROC AUC value: {}".format(roc_auc))
    rf_roc_auc = roc_auc_score(y_test, rf_pred_probs[:, 1])
    print("RF ROC AUC value:  {}".format(rf_roc_auc))
    # Output ROC of the chosen model using MCenter
    mlops.set_stat("XGBoost ROC AUC", roc_auc, st.TIME_SERIES)
    mlops.set_stat("RF ROC AUC", rf_roc_auc, st.TIME_SERIES)

    if roc_auc <= min_auc_requirement:
        mlops.health_alert("[Training] AUC Violation From Training Node",
                           "AUC Went Below {}. Current AUC Is {}".format(min_auc_requirement, roc_auc))

    # ROC curve
    fpr, tpr, thr = roc_curve(y_test, pred_probs[:, 1])
    rf_fpr, rf_tpr, rf_thr = roc_curve(y_test, rf_pred_probs[:, 1])

    cg = MultiGraph().name("Receiver Operating Characteristic ").set_continuous()
    cg.add_series(label='Random curve ''', x=fpr.tolist(), y=fpr.tolist())
    cg.add_series(label='XGBoost ROC curve (area = {0:0.2f})'''.format(roc_auc), x=fpr.tolist(), y=tpr.tolist())
    cg.add_series(label='RF ROC curve (area = {0:0.2f})'''.format(rf_roc_auc), x=rf_fpr.tolist(), y=rf_tpr.tolist())
    cg.x_title('False Positive Rate')
    cg.y_title('True Positive Rate')
    mlops.set_stat(cg)

    # Feature importance comparison
    # XGBoost Feature importance
    export_feature_importance(final_model, list(X_train.columns), 5, "XGBoost")
    export_feature_importance(rf_model, list(X_train.columns), 5, "RF")

    # KS Analysis
    max_pred_probs = pred_probs.max(axis=1)
    y_test0=np.where(y_test == 0)[0]
    y_test1=np.where(y_test == 1)[0]
    rf_max_pred_probs = rf_pred_probs.max(axis=1)

    # KS for the XGBoost model
    ks = ks_2samp(max_pred_probs[y_test0], max_pred_probs[y_test1])
    ks_stat = ks.statistic
    ks_pvalue = ks.pvalue
    print("KS values for XGBoost: \n Statistics: {} \n pValue: {}\n".format(ks_stat, ks_pvalue))
    # KS for the RF model
    rf_ks = ks_2samp(rf_max_pred_probs[y_test0], rf_max_pred_probs[y_test1])
    rf_ks_stat = rf_ks.statistic
    rf_ks_pvalue = rf_ks.pvalue
    print("RF KS values: \n Statistics: {} \n pValue: {}\n".format(rf_ks_stat, rf_ks_pvalue))
    # Output KS Stat of the chosen model using MCenter
    mlops.set_stat("KS Stats for CGBoost", ks_stat, st.TIME_SERIES)
    # Output KS Stat of the chosen model using MCenter
    mlops.set_stat("KS Stats for RF", rf_ks_stat, st.TIME_SERIES)

    # raising alert if ks-stat goes above required threshold
    if ks_stat >= max_ks_requirement:
        mlops.health_alert("[Training] KS Violation From Training Node",
                           "KS Stat Went Above {}. Current KS Stat Is {}".format(max_ks_requirement, ks_stat))

    ks_table = Table().name("KS Stats for XGBoost").cols(["Statistic", "pValue"])
    ks_table.add_row([ks_stat, ks_pvalue])
    mlops.set_stat(ks_table)

    # PSI Analysis
    # Calculating PSI
    total_psi, psi_table = get_psi(max_pred_probs[y_test0], max_pred_probs[y_test1])
    rf_total_psi, rf_psi_table = get_psi(rf_max_pred_probs[y_test0], rf_max_pred_probs[y_test1])
    psi_table_stat = Table().name("PSI Stats for XGBoost").cols(
        ["Base Pop", "Curr Pop", "Lower Bound", "Upper Bound", "Base Percent", "Curr Percent",
         "Segment PSI"])
    row_num = 1
    for each_value in psi_table.values:
        str_values = [str(i) for i in each_value]
        psi_table_stat.add_row(str(row_num), str_values)
        row_num += 1
    mlops.set_stat(psi_table_stat)
    print("Total XGBoost PSI values: \n {}".format(total_psi))
    #     Output Total PSI of the chosen model using MCenter
    mlops.set_stat("Total XGBoost PSI ", total_psi, st.TIME_SERIES)

    if total_psi >= min_psi_requirement:
        mlops.health_alert("[Training] PSI Violation From Training Node",
                           "PSI Went Below {}. Current PSI Is {}".format(min_psi_requirement,
                                                                         total_psi))

    print("Total RF PSI values: \n {}".format(rf_total_psi))
    rf_psi_table_stat = Table().name("PSI Stats for RF").cols(
        ["Base Pop", "Curr Pop", "Lower Bound", "Upper Bound", "Base Percent", "Curr Percent",
         "Segment PSI"])
    row_num = 1
    for each_value in rf_psi_table.values:
        str_values = [str(i) for i in each_value]
        rf_psi_table_stat.add_row(str(row_num), str_values)
        row_num += 1
    mlops.set_stat(rf_psi_table_stat)
    #     Output Total PSI of the chosen model using MCenter
    mlops.set_stat("Total RF PSI ", rf_total_psi, st.TIME_SERIES)

    # ## Save the XGBoost Model
    import pickle
    model_file = open(pm_options.output_model, 'wb')
    pickle.dump(final_model, model_file)
    model_file.close()

    # ## Finish the program
    mlops.done()


if __name__ == "__main__":
    main()
