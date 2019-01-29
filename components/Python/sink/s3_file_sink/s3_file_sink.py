from __future__ import print_function

import argparse
import os
import boto3
import uuid

from parallelm.components import ConnectableComponent
from parallelm.ml_engine.python_engine import PythonEngine


class S3FileSink(ConnectableComponent):

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        file_path = str(parent_data_objs[0])
        self._save_file(file_path)
        return []

    def _save_file(self, file_path):
        self._logger.info(" *** save file .. params:{}".format(self._params))

        client = boto3.client(
            's3',
            aws_access_key_id=self._params["aws_access_key_id"],
            aws_secret_access_key=self._params["aws_secret_access_key"],
        )
        data = open(file_path, 'rb')
        client.put_object(Bucket=self._params["bucket"], Key=self._params["key"], Body=data)

        return
