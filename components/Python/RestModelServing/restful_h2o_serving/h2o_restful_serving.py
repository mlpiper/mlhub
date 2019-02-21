from contextlib import closing
import glob
import logging
import os
import signal
import socket
import time
import subprocess

from py4j.java_gateway import JavaGateway, get_field
from py4j.java_gateway import GatewayParameters, CallbackServerParameters

from parallelm.common.mlcomp_exception import MLCompException
from parallelm.components.restful.flask_route import FlaskRoute
from parallelm.components.restful_component import RESTfulComponent
from parallelm.pipeline.components_desc import ComponentsDesc
from parallelm.pipeline.component_dir_helper import ComponentDirHelper


class H2oRESTfulServing(RESTfulComponent):

    JAVA_COMPONENT_ENTRY_POINT_CLASS = "com.parallelm.mlcomp.ComponentEntryPoint"
    JAVA_COMPONENT_CLASS_NAME = "com.parallelm.components.restful.H2oModelServing"

    class Java:
        implements = ["com.parallelm.mlcomp.MLOps"]

    def __init__(self, engine):
        super(H2oRESTfulServing, self).__init__(engine)

        self._verbose = self._logger.isEnabledFor(logging.DEBUG)
        self._gateway = None
        self._component_via_py4j = None
        self._model_loaded = False
        self._java_port = None
        self._prefix_msg = "wid: 0, "

    def post_fork_callback(self):
        self._prefix_msg = "wid: {}, ".format(self.get_wid())
        self._launch_custom_java_gateway()

    def _launch_custom_java_gateway(self):
        self._run_java_server_entry_point()
        self._setup_py4j_client_connection()

    def _run_java_server_entry_point(self):
        comp_realpath = os.path.realpath(__file__)
        comp_filename = os.path.basename(comp_realpath)
        comp_dirname = os.path.basename(os.path.dirname(comp_realpath))
        comp_module_name = "{}.{}.{}".format(
            ComponentsDesc.CODE_COMPONETS_MODULE_NAME,
            comp_dirname,
            os.path.splitext(comp_filename)[0])
        if self._verbose:
            self._logger.debug("comp_module_name: {}".format(comp_module_name))
        comp_helper = ComponentDirHelper(comp_module_name, comp_filename)
        comp_dir = comp_helper.extract_component_out_of_egg()
        if self._verbose:
            self._logger.debug(self._prefix_msg + "comp_dir: {}".format(comp_dir))

        jar_files = glob.glob(comp_dir + "/*.jar")
        java_cp = ":".join(jar_files)

        if self._verbose:
            self._logger.info(self._prefix_msg + "java_cp: {}".format(java_cp))

        self._java_port = H2oRESTfulServing.find_free_port()
        cmd = ["java", "-cp", java_cp, H2oRESTfulServing.JAVA_COMPONENT_ENTRY_POINT_CLASS, "--class-name",
               H2oRESTfulServing.JAVA_COMPONENT_CLASS_NAME, "--port", str(self._java_port)]
        if self._verbose:
            self._logger.debug(self._prefix_msg + "java gateway cmd: " + " ".join(cmd))

        self._proc = subprocess.Popen(cmd)  # , stdout=self._stdout_pipe_w, stderr=self._stderr_pipe_w)

        # TODO: provide a more robust way to check proper process startup
        time.sleep(2)

        poll_val = self._proc.poll()
        if poll_val is not None:
            raise Exception("java gateway failed to start")

        if self._verbose:
            self._logger.debug(self._prefix_msg + "java server entry point run successfully!")

    def _setup_py4j_client_connection(self):
        gateway_params = GatewayParameters(port=self._java_port,
                                           auto_field=True,
                                           auto_close=True,
                                           eager_load=True)
        callback_server_params = CallbackServerParameters(port=0,
                                                          daemonize=True,
                                                          daemonize_connections=True,
                                                          eager_load=True)
        self._gateway = JavaGateway(gateway_parameters=gateway_params,
                                    callback_server_parameters=callback_server_params,
                                    python_server_entry_point=self)
        self._component_via_py4j = self._gateway.entry_point.getComponent()
        if not self._component_via_py4j:
            raise MLCompException("None reference of py4j java object!")

        if self._verbose:
            self._logger.debug(self._prefix_msg + "Py4J component referenced successfully! comp_via_py4j: {}"
                               .format(self._component_via_py4j))

        self._component_via_py4j.setEnvAttributes(self.get_wid(), self._verbose)

    def configure_callback(self):
        if self._verbose:
            self._logger.debug(self._prefix_msg + "configure callback params: {}".format(self._params))
        if self._params:
            j_params = self._gateway.jvm.java.util.HashMap()

            for k, v in self._params.items():
                j_params[k] = v

            self._component_via_py4j.configure(j_params)
            if self._verbose:
                self._logger.debug(self._prefix_msg + "java restful class configured successfully!")

    def load_model_callback(self, model_path, stream, version):
        if self._verbose:
            self._logger.debug(self._prefix_msg + "load model callback, path: {}".format(model_path))

        if self._component_via_py4j:
            result = self._component_via_py4j.loadModel(model_path)
            if self._verbose:
                self._logger.debug(self._prefix_msg + "model loaded, result: {}, path: {}".format(result, model_path))
            self._model_loaded = True

    def setStat(self, stat_name, stat_value):
        """
        It is called by the java side, whenever a statistics needs to be set
        :param stat_name:  name of the statistics
        :param stat_value:  statistics value (int)
        """
        self._logger.info("Set stat, name: {}, value: {}".format(stat_name, stat_value))

    @FlaskRoute('/predict', raw=True)
    def predict(self, query_string, body_data):
        if self._verbose:
            self._logger.debug(self._prefix_msg + "predict, query_string: {}, body_data: {}".format(query_string, body_data))

        if self._model_loaded:
            result = self._component_via_py4j.predict(query_string, body_data)
            returned_code = get_field(result, "returned_code")
            json = get_field(result, "json")
            if self._verbose:
                self._logger.debug(self._prefix_msg + "got response ... code: {}, json: {}".format(returned_code, str(json)))
            return(returned_code, str(json))
        else:
            return 404, '{"error": "H2O model was not loaded yet!"}'

    def cleanup_callback(self):
        """
        The cleanup function is called when the process exists
        """
        if self._verbose:
            self._logger.debug(self._prefix_msg + "cleaning up RESTful component ...")

        if self._gateway:
            try:
                if self._verbose:
                    self._logger.debug(self._prefix_msg + "shutting down gateway ...")
                self._gateway.shutdown()
            except Exception as ex:
                self._logger.info(self._prefix_msg + "exception in gateway shutdown, {}".format(ex))

        if self._proc:
            if self._verbose:
                self._logger.debug(self._prefix_msg + "killing gateway server ...")
            os.kill(self._proc.pid, signal.SIGTERM)
            os.kill(self._proc.pid, signal.SIGKILL)

    @staticmethod
    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]


if __name__ == '__main__':
    import argparse

    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to listen on")
    parser.add_argument("input_model", help="Path of input model to create")
    parser.add_argument("--log_level", choices=log_levels.keys(), default="info", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s %(levelname)s [%(module)s:%(lineno)d]:  %(message)s')
    logging.getLogger().setLevel(log_levels[args.log_level])

    comp_module_name = os.path.splitext(os.path.basename(__file__))[0]
    ComponentsDesc.CODE_COMPONETS_MODULE_NAME = comp_module_name

    H2oRESTfulServing.run(port=args.port, model_path=args.input_model)
