from __future__ import print_function

from parallelm.mlops import mlops as mlops
import numpy as np
import pandas as pd
import argparse
import random
import pickle

from parallelm.mlops import StatCategory as st
from parallelm.mlops.predefined_stats import PredefinedStats
from parallelm.mlops.stats.bar_graph import BarGraph

def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", help="Path of input model to create")
    options = parser.parse_args()
    return options


def main():
    pm_options = parse_args()
    mlops.init() 
    # Load the model
    if pm_options.input_model is not None:
        try:
            filename = pm_options.input_model
            file_obj = open(filename, 'rb')
            mlops.set_stat("model_file", 1)
        except Exception as e:
            print("Model not found")
            print("Got exception: {}".format(e))
            mlops.set_stat("model_file", 0)
            mlops.done()
            return 0

    classifier = pickle.load(file_obj)

    # Create synthetic data (Gaussian Distribution, Poisson Distribution and Beta Distribution)
    num_samples = 50
    num_features = 20

    np.random.seed(0)
    g = np.random.normal(0, 1, (num_samples,num_features))
    p = np.random.poisson(0.7, (num_samples,num_features))
    b = np.random.beta(2, 2, (num_samples,num_features))

    test_data = np.concatenate((g, p, b), axis=0)
    np.random.seed()
    test_features = test_data[np.random.choice(test_data.shape[0], num_samples, replace=False)]

    ########## Start of ParallelM instrumentation ##############
    '''
    MLOps API to report the distribution statistics of each feature in the data and compare it automatically with the ones 
    reported during training to generate the similarity score.
    '''    
    mlops.set_data_distribution_stat(test_features)
    ########## End of ParallelM instrumentation ##############

    ########## Start of ParallelM instrumentation ##############
    # Report the number of samples being processed
    mlops.set_stat(PredefinedStats.PREDICTIONS_COUNT, num_samples, st.TIME_SERIES)
    ########## End of ParallelM instrumentation ##############

    # Predict labels
    result = classifier.predict(test_features)
    # Predict probability
    class_probability = classifier.predict_proba(test_features)
    maximum_prob = np.max(class_probability,axis=1)

    # Label distribution in training
    value, counts = np.unique(result, return_counts=True)
    label_distribution = np.asarray((value, counts)).T
    column_names = value.astype(str).tolist()
    print("Label distributions: \n {0}".format(label_distribution))

    ########## Start of ParallelM instrumentation ##############
    # Report label distribution as a BarGraph
    bar = BarGraph().name("Label Distribution").cols((label_distribution[:,0]).astype(str).tolist()).data((label_distribution[:,1]).tolist())
    mlops.set_stat(bar)
    ########## End of ParallelM instrumentation ##############    


    mlops.done()


if __name__ == "__main__":
    main()


