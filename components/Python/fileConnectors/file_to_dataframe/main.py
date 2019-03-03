from __future__ import print_function

import argparse
import sys
import time
import os
import pandas

from parallelm.components import ConnectableComponent
from parallelm.mlops.stats.multi_line_graph import MultiLineGraph
from parallelm.mlops import mlops as mlops

class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for read_file_to_df 
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        file_path = str(parent_data_objs[0])
        if file_path is None:
            file_path = self._params.get('file_path')
        return [read_file_to_df(file_path)]


def read_file_to_df(filepath):
    """
    Read file and return DataFrame
    """
    mlops.init()
    if not os.path.exists(filepath):
        print("stderr- failed to find {}".format(filepath), file=sys.stderr)
        raise Exception("file path does not exist: {}".format(filepath))

    test_data = pandas.read_csv(filepath)
    mlops.done()
    return test_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", default='/tmp/test-data.csv', help="Dataset to read")
    options = parser.parse_args()
    return options

