from __future__ import print_function

import argparse
import sys
import time
import pprint
import os
import json
import requests
from collections import namedtuple
import boto3
import uuid

from parallelm.components import ConnectableComponent
from parallelm.common import model_connector_constants
from parallelm.common.external_model_info import ExternalModelInfo
from parallelm.common.model_connector_helper import get_options_from_env_json_info, model_connector_mode
from parallelm.mlops.models.model import ModelFormat


def vprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for the do_train
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        options = namedtuple("Options", self._params.keys())(*self._params.values())
        model_info = import_model(options)
        return [model_info]


def _get_tmp_dir():
    return os.environ.get(model_connector_constants.TMP_DIR_ENV, "/tmp")


def import_model(options):
    vprint(" *** import_model.. options:{}".format(options))

    client = boto3.client(
        's3',
        aws_access_key_id=options.aws_access_key_id,
        aws_secret_access_key=options.aws_secret_access_key,
    )

    model_path = os.path.join(_get_tmp_dir(), "s3_model_" + str(uuid.uuid4()))
    client.download_file(options.bucket, options.key, model_path)

    description = "S3:\n bucket: {}\nkeu: {}"\
        .format(options.bucket, options.key)
    model_info = ExternalModelInfo(path=model_path, format=ModelFormat.BINARY, descriptipn=description)
    return model_info


def list_params(options):
    raise Exception("Not done yet")


def parse_args():

    parser = argparse.ArgumentParser()

    # The cmd is used to determine how to run the connector (import/list_params)
    parser.add_argument("--" + model_connector_constants.MODEL_CONNECTOR_CMD_OPTION,
                        default=model_connector_constants.MODEL_CONNECTOR_IMPORT_CMD,
                        help="command to perform: {}, {}".
                        format(model_connector_constants.MODEL_CONNECTOR_IMPORT_CMD,
                               model_connector_constants.MODEL_CONNECTOR_LIST_PARAM_CMD))

    # All arguments below are components arguments
    parser.add_argument("--aws-access-key-id", default=None, help="Access key ID")
    parser.add_argument("--aws-secret-access-key", default=None, help="Secret key")
    parser.add_argument("--region", default=None, help="AWS region name")
    parser.add_argument("--bucket", default=None, help="S3 bucket name")
    parser.add_argument("--key", default=None, help="S3 key name")
    options = parser.parse_args()
    return options


def main():
    options = parse_args()

    if model_connector_mode():
        options = get_options_from_env_json_info(options)

    if options.cmd == model_connector_constants.MODEL_CONNECTOR_IMPORT_CMD:
        model_info = import_model(options)
        print(model_info.to_json())
    elif options.cmd == model_connector_constants.MODEL_CONNECTOR_LIST_PARAM_CMD:
        list_params(options)
    else:
        raise Exception("CMD: [{}] is not supported".format(options.cmd))


if __name__ == "__main__":
    main()
