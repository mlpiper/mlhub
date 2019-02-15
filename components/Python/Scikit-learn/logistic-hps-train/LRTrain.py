import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import argparse
import sys

from parallelm.mlops import mlops as mlops
from parallelm.mlops import StatCategory as st
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.table import Table
from sklearn.model_selection import cross_val_score

def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", help="Data file to use as input")
    parser.add_argument("--output-model", help="Data file to save model")
    parser.add_argument("--regularization-range", help="Regularization HPS list",
                        default="0.001,0.01,0.1,1.0,10,100", required=False)
    options = parser.parse_args()
    return options


def main():
    pm_options = parse_args()
    print("PM: Configuration:")
    print("PM: Data file:            [{}]".format(pm_options.data_file))
    print("PM: Output model:         [{}]".format(pm_options.output_model))
    print("PM: regularization_range:         [{}]".format(pm_options.regularization_range))

    mlops.init()

    # Read the Samsung datafile
    dataset = pd.read_csv(pm_options.data_file)

    # Separate into features and labels
    features = dataset.iloc[:, 1:].values
    labels = dataset.iloc[:, 0].values

    # Hyper-parameter search using k-fold cross-validation
    # Applying k_fold cross validation
    regularization_range = pm_options.regularization_range.split(',')
    regularization = [float(regularization_var) for regularization_var in regularization_range]
    tune_parameters = [{'C': regularization}]

    # Initialize logistic regression algorithm
    LR = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='lbfgs')
    clf = GridSearchCV(LR, tune_parameters, cv=5,
                       scoring='accuracy')
    clf.fit(features, labels)
    print("best parameter = ", clf.best_params_)
    accuracy = clf.cv_results_['mean_test_score']
    print('Accuracy values: \n {0} \n for `Regularization values: \n{1}'.format(accuracy, regularization))

    ########## Start of ParallelM instrumentation ##############
    # Report Hyper-parameter Table
    tbl = Table().name("Hyper-parameter Search Results").cols(["Mean accuracy from k-fold cross-validation"])
    print("length of regularization",len(regularization))
    index_max = np.argmax(accuracy)
    for a in range(0,len(regularization)):
        print("adding row", regularization[a])
        if a == index_max:
            tbl.add_row("[Best] Regularization = " + np.str(regularization[a]),[accuracy[a]])
        else:
            tbl.add_row("Regularization = " + np.str(regularization[a]),[accuracy[a]])
    mlops.set_stat(tbl)
    ########## End of ParallelM instrumentation ##############


    # Label distribution in training
    label_distribution = dataset['label'].value_counts()
    column_names = np.array(label_distribution.index).astype(str).tolist()
    print("Label distributions: \n {0}".format(label_distribution))

    ########## Start of ParallelM instrumentation ##############
    # Report label distribution as a BarGraph
    bar = BarGraph().name("Label Distribution").cols(np.array(label_distribution.index).astype(str).tolist()).data(label_distribution.values.tolist())
    mlops.set_stat(bar)
    ########## Start of ParallelM instrumentation ##############


    #################### Start of ParallelM instrumentation ################
    # Report accuracy of the chosen model
    mlops.set_stat("K-fold cross-validation Accuracy", accuracy[index_max], st.TIME_SERIES)
    #################### End of ParallelM instrumentation ################

    # Histogram input
    mlops.set_data_distribution_stat(dataset)

    # Save the model
    import pickle
    model_file =  open(pm_options.output_model, 'wb')
    pickle.dump(clf, model_file)
    model_file.close()
    mlops.done()

if __name__ == "__main__":
    main()
