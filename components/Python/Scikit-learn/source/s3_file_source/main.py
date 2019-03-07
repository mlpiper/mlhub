from __future__ import print_function

import argparse
import os
import boto3
import uuid
import time
from itertools import islice

from parallelm.components import ConnectableComponent
from parallelm.ml_engine.python_engine import PythonEngine
from parallelm.mlops import mlops as mlops

class S3FileSource(ConnectableComponent):

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):

        file_path = self._fetch_file()
        return [file_path]

    def _fetch_file(self):
        file_size = line_count = 0

        # Initialize mlops
        mlops.init()

        client = boto3.client(
            's3',
            aws_access_key_id=self._params["aws_access_key_id"],
            aws_secret_access_key=self._params["aws_secret_access_key"],
        )

        if self._params["get_file_size"]:
            resp_obj = client.head_object(Bucket=self._params["bucket"], Key=self._params["key"])
            file_size = resp_obj['ContentLength'] / (1024 * 1024)
            mlops.set_stat("s3.inputFileSizeMB", file_size)

        file_path = os.path.join(self._params["parent_directory"], "s3_file_" + str(uuid.uuid4()))

        fetch_start_time = time.time()
        client.download_file(self._params["bucket"], self._params["key"], file_path)
        fetch_elapsed_time = time.time() - fetch_start_time
        if self._params["get_fetch_time"]:
            mlops.set_stat("s3.inputFetchTimemsec", fetch_elapsed_time)

        # get line-count for the file (loads file in memory)
        # should help keep prediction latency NOT be IO-bound
        if self._params["get_line_count"]:
            line_count = len(open(file_path).readlines())
            mlops.set_stat("s3.inputFileLineCount", line_count)

        return file_path
