from __future__ import print_function

from parallelm.mlops import mlops as mlops
import numpy as np
import pandas as pd
import argparse
from random import *
import pickle

from parallelm.mlops import StatCategory as st
from parallelm.mlops.stats.bar_graph import BarGraph

def parse_args():
    """
    Parse Arguments from component
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", help="Path of input model to create")
    parser.add_argument("--input-file", help = "Path of the file to process")
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

    # Load the data
    test_dataset = pd.read_csv(pm_options.input_file)
    
    mlops.set_data_distribution_stat(test_dataset)
    # Extract numpy array
    test_features = test_dataset.values
    # Predict labels
    result = classifier.predict(test_features)
    # Predict probability
    class_probability = classifier.predict_proba(test_features)
    maximum_prob = np.max(class_probability,axis=1)

    # Tag samples that are below a certain probability and write to a file
    confidence = 0.8
    low_prob_samples = test_features[np.where(maximum_prob < confidence)]
    low_prob_predictions = result[np.where(maximum_prob < confidence)]
    unique_elements_low, counts_elements_low = np.unique(low_prob_predictions, return_counts=True)
    unique_elements_low = [str(i) for i in unique_elements_low]
    print("Low confidence predictions: \n {0} \n with frequency {1}".format(unique_elements_low, counts_elements_low))

    ########## Start of ParallelM instrumentation ##############
    # BarGraph showing distribution of low confidence labels
    bar = BarGraph().name("Low confidence label distribution").cols(unique_elements_low).data(counts_elements_low.tolist())
    mlops.set_stat(bar)
    ########## End of ParallelM instrumentation ################

    # Samples with high probability
    high_prob_samples = test_features[np.where(maximum_prob > confidence)]
    high_prob_predictions = result[np.where(maximum_prob > confidence)]
    unique_elements_high, counts_elements_high = np.unique(high_prob_predictions, return_counts=True)
    unique_elements_high = [str(i) for i in unique_elements_high]
    print("High confidence predictions: \n {0} \n with frequency {1}".format(unique_elements_high, counts_elements_low))

    ########## Start of ParallelM instrumentation ##############
    # BarGraph showing distribution of high confidence labels
    bar = BarGraph().name("High confidence label distribution").cols(unique_elements_high).data(counts_elements_high.tolist())
    mlops.set_stat(bar)
    ########## End of ParallelM instrumentation ################

    mlops.done()


if __name__ == "__main__":
    main()


