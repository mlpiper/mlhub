from __future__ import print_function

import argparse
import sys
import time
import pprint
import os
import json
import requests
import zipfile
from configparser import ConfigParser

from collections import namedtuple
from collections import OrderedDict

from parallelm.components import ConnectableComponent
from parallelm.common import model_connector_constants
from parallelm.mlops.models.model import ModelFormat
from parallelm.common.external_model_info import ExternalModelInfo
from parallelm.common.model_connector_helper import get_options_from_env_json_info, model_connector_mode


def vprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class ComponentH2oModelConnector(ConnectableComponent):
    """
    Adapter for the import_model
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        options = namedtuple("Options", self._params.keys())(*self._params.values())
        model_info = import_model(options)
        return [model_info]


def _get_tmp_dir():
    return os.environ.get(model_connector_constants.TMP_DIR_ENV, "/tmp")


class H2OClient:
    """
    Defines interfaces that can be invoked on the h2o ver 3.22.x
    """
    def __init__(self, server_url, api_key=None):

        if not server_url:
            raise Exception("Host detail is missing")

        self._server_url = server_url
        self._api_key = api_key

    def list_models(self, context=None):
        """
        Interface to list all models from the H2o.ai server
        """
        url = "/".join([self._server_url, "3/Models"])
        r = requests.get(
            url,
            headers={"Content-Type": "application/json"},
            auth=(self._api_key, ""),
            verify=False
        )
        if r.status_code != requests.codes.ok:
            vprint("Bad request:{} {}".format(r.status_code, r.raise_for_status()))
            return None
        return r.json()

    def download_file(self, model_name, out_path="/tmp/"):
        """
        Download MOJO model from the H2o.ai server, known "model_name" provided
        """
        url = "/".join([self._server_url, "3/Models", model_name, "mojo"])
        r = requests.get(url, stream=True, auth=(self._api_key, ""), verify=False)
        if r.status_code != requests.codes.ok:
            vprint("Bad request:{} {}".format(r.status_code, r.raise_for_status()))
            return None
        save_file_path = os.path.join(out_path, model_name)
        with open(save_file_path, 'wb') as f:
            # use-case of a large model, using HTTP to stream the data where in the data
            # can be received in chunks, optimizing for memory over time
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return save_file_path

    def extract_model_info(self, model_path):

        archive = zipfile.ZipFile(model_path, 'r')
        model_info_str = str(archive.read('model.ini'), 'utf-8')
        config = ConfigParser(allow_no_value=True)
        config.read_string(model_info_str)
        model_info = OrderedDict()
        model_info["algorithm"] = config["info"]["algorithm"]
        model_info["h2o_version"] = config["info"]["h2o_version"]
        return model_info


def import_model(options):
    vprint("H2o.ai import_model.. options:{}".format(options))
    h2oclient = H2OClient(server_url=options.server_url)

    model_path = h2oclient.download_file(model_name=options.model_name,
                                         out_path=_get_tmp_dir())
    h2o_model_info = h2oclient.extract_model_info(model_path)
    h2o_model_info["file"] = os.path.basename(model_path)
    h2o_description = json.dumps(h2o_model_info)
    model_info = ExternalModelInfo(path=model_path, format=ModelFormat.H2O_3, descriptipn=h2o_description)
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
    parser.add_argument("--server-url", default="localhost", help="Server URL")
    parser.add_argument("--model-name", default=None, help="Model name")
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
