from __future__ import print_function

import argparse
import sys
import time
import os
import pickle
import pandas as pd
import numpy as np

from sklearn.exceptions import NotFittedError
from parallelm.components import ConnectableComponent
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph
from parallelm.mlops.stats.multi_line_graph import MultiLineGraph
from parallelm.mlops.predefined_stats import PredefinedStats


class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for the do_predict
    """
    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        df_infer_data = parent_data_objs[0]
        input_model = self._params.get("input-model")
        return[do_predict(df_infer_data, input_model)]


def do_predict(df_infer, input_model):
    """
    Perform predictions:
        a) load scikit-learn model
        b) on dataset run predict, collect stats assocaited to predictions
        c) obtain class probability stats
        d) report stats using mlops APIs
    """
    prog_start_time = time.time()
    mlops.init()
    model = pickle.load(open(input_model, "rb"))
    data_features = df_infer.values

    # Start timer (inference)
    inference_start_time = time.time()
    # Predict labels
    predict_results = model.predict(df_infer)
    # End timer (inference)
    inference_elapsed_time = time.time() - inference_start_time

    # Predict probability
    class_probability = model.predict_proba(df_infer)
    maximum_prob = np.max(class_probability, axis=1)
    # Tag samples that are below a certain probability and write to a file
    confidence = 0.7

    low_prob_predictions = predict_results[np.where(maximum_prob < confidence)]
    unique_elements_low, counts_elements_low = np.unique(low_prob_predictions, return_counts=True)
    unique_elements_low = [str(i) for i in unique_elements_low]
    # self._logger.info("Low confidence predictions: \n {0} \n with frequency {1}".format(unique_elements_low, counts_elements_low))

    # ########## Start of MCenter instrumentation ##############
    # # BarGraph showing distribution of low confidence labels
    bar = BarGraph().name("Low confidence label distribution").cols(unique_elements_low).data(counts_elements_low.tolist())
    # self._logger.info("Low bar : ", type(bar), "->", bar)
    mlops.set_stat(bar)

    # ########## End of MCenter instrumentation ################
    #
    # # Samples with high probability
    high_prob_predictions = predict_results[np.where(maximum_prob > confidence)]
    unique_elements_high, counts_elements_high = np.unique(high_prob_predictions, return_counts=True)
    unique_elements_high = [str(i) for i in unique_elements_high]
    # self._logger.info("High confidence predictions: \n {0} \n with frequency {1}".format(unique_elements_high,
    #                                                                                 counts_elements_low))

    # ########## Start of MCenter instrumentation ##############
    # #  BarGraph showing distribution of high confidence labels
    bar = BarGraph().name("High confidence label distribution").cols(unique_elements_high).data(counts_elements_high.tolist())
    # self._logger.info("High bar : ", type(bar), "->", bar)
    mlops.set_stat(bar)
    ########## End of MCenter instrumentation ################

    ########  Report PM stats ###########
    mlops.set_stat(PredefinedStats.PREDICTIONS_COUNT, len(data_features))
    prog_elapsed_time = time.time() - prog_start_time
    mlt_time = MultiLineGraph().name("Time Elapsed").labels(["Program Time", "Inference Time"])
    mlt_time.data([prog_elapsed_time, inference_elapsed_time])
    mlops.set_stat(mlt_time)
    ###  End of PM stats reporting #####

    mlops.done()

    # return predict_results
    return class_probability


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", help="Path to load model from")
    options = parser.parse_args()
    return options

