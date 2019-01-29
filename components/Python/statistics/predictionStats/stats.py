from __future__ import print_function

import argparse
import os
import boto3
import uuid

import numpy as np
import pandas as pd

from parallelm.components import ConnectableComponent
from parallelm.mlops import mlops as mlops
from parallelm.mlops.stats.bar_graph import BarGraph


class StatsReport(ConnectableComponent):

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        file_path = str(parent_data_objs[0])
        self._report_stats(file_path)
        return []

    def _report_stats(self, file_path):
        self._logger.info(
            " *** generate stats .. params:{}".format(self._params))
        self._logger.info(" *** Source file {}".format(file_path))

        # Read the file
        data = pd.read_csv(file_path, sep=' |,', header=None, skiprows=1)
        data = data.rename(
            index=str,
            columns={
                1: "label",
                2: "confidence0",
                3: "confidence1"})
        prediction_distribution = data['label'].value_counts()
        column_names = np.array(
            prediction_distribution.index).astype(str).tolist()

        # Initialize mlops
        mlops.init()

        # Report a bar graph
        bar = BarGraph().name("Prediction Distribution").cols(
            np.array(
                prediction_distribution.index).astype(str).tolist()).data(
            prediction_distribution.values.tolist())
        mlops.set_stat(bar)

        # Generate an alert on low confidence if the argument is set to true
        if (self._params["alert"]):
            index = data.values[:, 1].astype(int)
            confidence = data.values[:, 2:4]
            confidence_per_prediction = confidence[:, index][:, 0] * 100
            low_conf_percent = len(
                confidence_per_prediction[confidence_per_prediction < self._params["confidence"]]) / len(confidence_per_prediction) * 100
            if low_conf_percent > self._params["samples"]:
                msg = "Low confidence: {}% of inferences had confidence below {}%".format(
                    low_conf_percent, self._params["confidence"])
                print(msg)
                mlops.health_alert("Low confidence alert", msg)

        mlops.done()

        return []
