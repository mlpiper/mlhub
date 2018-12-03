import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
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
    parser.add_argument("--max_iter", dest="max_iter", type=int, required=False, default=100, help='Maximum number of iterations')
    parser.add_argument("--output-model", help="Data file to save model")
    options = parser.parse_args()
    return options

def main():
    pm_options = parse_args()
    print("PM: Configuration:")
    print("PM: Output model:         [{}]".format(pm_options.output_model))

    mlops.init()

    # Create sythetic data using scikit learn
    X,y = make_classification(n_samples=50, n_features=20, n_informative=2, n_redundant=1, n_classes=3, n_clusters_per_class=1, random_state=42)
                           

    # Separate into features and labels
    features = X
    labels = y

    # Add noise to the data
    noisy_features = np.random.uniform(0, 10) * np.random.normal(0, 1, (50,20))
    features = features + noisy_features

    # Create a model that should be deployed into production
    final_model = LogisticRegression(class_weight='balanced', multi_class='multinomial', max_iter=pm_options.max_iter, solver='lbfgs')
    final_model.fit(features,labels)

    # Accuracy for the chosen model
    accuracy = final_model.score(features, labels)
    print("Accuracy values: \n {0}".format(accuracy))

    # Label distribution in training
    value, counts = np.unique(labels, return_counts=True)
    label_distribution = np.asarray((value, counts)).T
    column_names = value.astype(str).tolist()
    print("Label distributions: \n {0}".format(label_distribution))

    ########## Start of ParallelM instrumentation ##############
    # Report label distribution as a BarGraph
    bar = BarGraph().name("Label Distribution").cols((label_distribution[:,0]).astype(str).tolist()).data((label_distribution[:,1]).tolist())
    mlops.set_stat(bar)
    ########## End of ParallelM instrumentation ##############


    #################### Start of ParallelM instrumentation ################
    # Report accuracy of the chosen model
    mlops.set_stat("Accuracy", accuracy, st.TIME_SERIES)
    #################### End of ParallelM instrumentation ################

    #################### Start of ParallelM instrumentation ################
    # Histogram input
    '''
    MLOps API to report the distribution statistics of each feature in the data
    ''' 
    mlops.set_data_distribution_stat(features)
    #################### End of ParallelM instrumentation ################

    # Save the model
    import pickle
    model_file =  open(pm_options.output_model, 'wb')
    pickle.dump(final_model, model_file)
    model_file.close()
    mlops.done()

if __name__ == "__main__":
    main()
