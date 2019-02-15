from __future__ import print_function

import argparse
import os
import json
import requests
import re
import sys
from collections import namedtuple

from parallelm.components import ConnectableComponent
from parallelm.common import model_connector_constants
from parallelm.mlops.models.model import ModelFormat
from parallelm.common.external_model_info import ExternalModelInfo
from parallelm.common.model_connector_helper import get_options_from_env_json_info, model_connector_mode


def vprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class MCenterComponentAdapter(ConnectableComponent):
    """
    Adapter for the do_train
    """

    def __init__(self, engine):
        super(self.__class__, self).__init__(engine)

    def _materialize(self, parent_data_objs, user_data):
        model_info = import_model(self._params)
        return [model_info]


def _get_tmp_dir():
    return os.environ.get(model_connector_constants.TMP_DIR_ENV, "/tmp")


def is_downloadable(header):
    """
    Does the url contain a downloadable resource
    """

    content_type = header.get('content-type')
    if 'text' in content_type.lower():
        return False
    if 'html' in content_type.lower():
        return False
    return True


def get_size(header):
    content_length = header.get('content-length', None)
    if not content_length:
        return -1
    return int(content_length)


def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]


def import_model(options):
    url = options.url
    vprint("URL:[{}]".format(url))

    h = requests.head(url, allow_redirects=True, verify=False)
    header = h.headers

    # TODO: add flag to component description
    if not is_downloadable(header):
        raise Exception("URL is not downloadable")

    # TODO: add ability to limit by maximal size
    size = get_size(header)
    vprint("size of Model: {}".format(size))

    r = requests.get(url, allow_redirects=True, verify=False)
    cd = r.headers.get('content-disposition')

    if cd:
        filename = get_filename_from_cd(cd)
    elif url.find('/'):
        filename = url.rsplit('/', 1)[1]
    else:
        filename = "model.binary"

    vprint("file_name: {}".format(filename))
    filename = os.path.basename(filename)
    model_path = os.path.join(_get_tmp_dir(), filename)
    vprint("file_name: {}".format(model_path))
    open(model_path, 'wb').write(r.content)

    model_info = ExternalModelInfo(path=model_path, format=ModelFormat.BINARY, descriptipn=url)
    return model_info


def list_params(options):
    return {}


def parse_args():

    parser = argparse.ArgumentParser()

    # The cmd is used to determine how to run the connector (import/list_params)
    parser.add_argument("--" + model_connector_constants.MODEL_CONNECTOR_CMD_OPTION,
                        default=model_connector_constants.MODEL_CONNECTOR_IMPORT_CMD,
                        help="command to perform: {}, {}".
                        format(model_connector_constants.MODEL_CONNECTOR_IMPORT_CMD,
                               model_connector_constants.MODEL_CONNECTOR_LIST_PARAM_CMD))

    # All arguments below are components arguments
    parser.add_argument("--url", default=None, help="URL to import model from")

    options = parser.parse_args()
    return options


def main():
    options = parse_args()

    # If we detect the environment which contains the config we take it from there.
    # Only the CMD is taken from command line
    if model_connector_mode():
        options = get_options_from_env_json_info(options)

    if options.cmd == model_connector_constants.MODEL_CONNECTOR_IMPORT_CMD:
        model_info = import_model(options)
        print(model_info.to_json())
    elif options.cmd == model_connector_constants.MODEL_CONNECTOR_LIST_PARAM_CMD:
        list_info = list_params(options)
        print(json.dumps(list_info, indent=0))
    else:
        raise Exception("CMD: [{}] is not supported".format(options.cmd))


if __name__ == "__main__":
    main()
