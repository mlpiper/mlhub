from __future__ import print_function

import argparse
import os
import boto3
import uuid
import time

from parallelm.components import ConnectableComponent
from parallelm.ml_engine.python_engine import PythonEngine
from parallelm.mlops import mlops as mlops


class S3FileSink(ConnectableComponent):

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        file_path = str(parent_data_objs[0])
        self._save_file(file_path)
        return []

    def _save_file(self, file_path):

        # Initialize mlops
        mlops.init()

        if self._params["get_save_file_size"]:
            file_size = os.stat(file_path).st_size / (1024 * 1024)
            mlops.set_stat("s3.outputFileSizeMB", file_size)

        if self._params["get_save_line_count"]:
            line_count = len(open(file_path).readlines())
            mlops.set_stat("s3.outputFileLineCount", line_count)

        client = boto3.client(
            's3',
            aws_access_key_id=self._params["aws_access_key_id"],
            aws_secret_access_key=self._params["aws_secret_access_key"],
        )

        data = open(file_path, 'rb')
        save_start_time = time.time()
        client.put_object(Bucket=self._params["bucket"], Key=self._params["key"], Body=data)
        save_elapsed_time = time.time() - save_start_time
        if self._params["get_save_time"]:
            mlops.set_stat("s3.outputSaveTimemsec", save_elapsed_time)

        return
