from __future__ import print_function

import argparse
import sys
import time
import pprint
import os
import json
import requests
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
        options = namedtuple("Options", self._params.keys())(*self._params.values())
        model_info = import_model(options)
        return [model_info]


def _get_tmp_dir():
    return os.environ.get(model_connector_constants.TMP_DIR_ENV, "/tmp")


class CDSWClient:
    """
    Defines interfaces that can be invoked on the CDSW ver 1.4.2
    """
    def __init__(self, server_url, user, api_key):

        if not server_url or not user or not api_key:
            raise Exception("host, user or api key are missing")

        self._server_url = server_url
        self._user = user
        self._api_key = api_key

    def list_projects(self, context):
        url = "/".join([self._server_url, "api/v1/users", context, "projects"])
        res = requests.get(
            url,
            headers={"Content-Type": "application/json"},
            auth=(self._api_key, ""),
            verify=False
        )
        return 0, res.json()

    def execute_run(self, project_id, script, arguments):
        url = "/".join([self._server_url, "api/altus-ds-1/ds/run"])
        job_params = {"project": project_id, "script": script, "arguments": arguments}
        res = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            auth=(self._api_key, ""),
            data=json.dumps(job_params)
        )
        return 0, res.json()

    def list_runs(self, project_id):
        url = "/".join([self._server_url, "api/altus-ds-1/ds/listruns"])
        job_params = {"project": project_id, "pageSize": 30, "metricsOrder": "auroc", "orderSort": "desc"}
        res = requests.post(
            url,
            headers = {"Content-Type": "application/json"},
            auth=(self._api_key, ""),
            data=json.dumps(job_params),
            verify=False
        )
        return 0, res.json()

    def promote_output(self, expirement_id, file):
        url = "/".join([self._server_url, "api/altus-ds-1/ds/promoteRunOutput"])
        job_params = {"id":expirement_id, "files":[file]}
        res = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            auth=(self._api_key, ""),
            data=json.dumps(job_params),
            verify=False
        )
        return res.status_code

    def download_file(self, promoted, file, out_path="/tmp/"):
        url = "/".join([self._server_url, "api/v1/projects", promoted, "files", file])
        r = requests.get(url, stream=True, auth = (self._api_key, ""), verify=False)
        if r.status_code != requests.codes.ok:
            vprint("Bad request:{} {}".format(r.status_code, r.raise_for_status()))
            return None
        save_file_path = os.path.join(out_path, file)
        with open(save_file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return save_file_path


def import_model(options):
    vprint(" *** import_model.. options:{}".format(options))
    cdsw = CDSWClient(server_url=options.server_url, user=options.user, api_key=options.api_key)

    model_path = cdsw.download_file(promoted=options.promoted_path,
                                    file=options.model_name,
                                    out_path=_get_tmp_dir())

    cdsw_description = "CDSW:\n user: {}\nproject: {}\n model: {}"\
        .format(options.user, options.promoted_path, options.model_name)
    model_info = ExternalModelInfo(path=model_path, format=ModelFormat.BINARY, descriptipn=cdsw_description)
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
    parser.add_argument("--user", default=None, help="Username")
    parser.add_argument("--api-key", default=None, help="API Key")
    parser.add_argument("--promoted-path", default=None, help="Promoted model path")
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
