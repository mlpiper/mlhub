from __future__ import print_function

import argparse
import os
import boto3
import uuid

from parallelm.components import ConnectableComponent
from parallelm.ml_engine.python_engine import PythonEngine

class S3FileSource(ConnectableComponent):

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):

        file_path = self._fetch_file()
        return [file_path]

    def _fetch_file(self):
        self._logger.info(" *** import_model.. params:{}".format(self._params))

        client = boto3.client(
            's3',
            aws_access_key_id=self._params["aws_access_key_id"],
            aws_secret_access_key=self._params["aws_secret_access_key"],
        )

        file_path = os.path.join(self._params["parent_directory"], "s3_file_" + str(uuid.uuid4()))
        client.download_file(self._params["bucket"], self._params["key"], file_path)

        return file_path
