import logging
import numpy as np
import pickle
import sys

from parallelm.components.restful.flask_route import FlaskRoute
from parallelm.components.restful_component import RESTfulComponent


class SklearnRESTfulServing(RESTfulComponent):
    JSON_KEY_NAME = "prediction_vector"

    def __init__(self, engine):
        super(SklearnRESTfulServing, self).__init__(engine)
        self._classifier = None
        self._verbose = self._logger.isEnabledFor(logging.DEBUG)

    def load_model_callback(self, model_path, stream, version):
        self._logger.info("Model is loading, wid: {}, path: {}".format(self.get_wid(), model_path))
        with open(model_path, "rb") as f:
            self._classifier = pickle.load(f) if sys.version_info[0] < 3 \
                else self._classifier = pickle.load(f, encoding='latin1')

            if self._verbose:
                self._logger.debug("Un-pickled model: {}".format(self._classifier))
            self._logger.debug("Model loaded successfully!")

    @FlaskRoute('/predict')
    def predict(self, url_params, form_params):
        if SklearnRESTfulServing.JSON_KEY_NAME not in form_params:
            msg = "Unexpected json format for prediction! Missing '{}' key in: {}" \
                .format(SklearnRESTfulServing.JSON_KEY_NAME, form_params)
            self._logger.error(msg)
            raise Exception(msg)

        if self._verbose:
            self._logger.debug("predict, url_params: {}, form_params: {}".format(url_params, form_params))

        if not self._classifier:
            return (404, {"error": "Model not loaded yet!"})
        else:
            if self._verbose:
                self._logger.debug("type<form_params>: {}\n{}".format(type(form_params), form_params))
                two_dim_array = np.array([form_params[SklearnRESTfulServing.JSON_KEY_NAME]])
                self._logger.debug("type(two_dim_array): {}\n{}".format(type(two_dim_array), two_dim_array))
                prediction = self._classifier.predict(two_dim_array)
                self._logger.debug("prediction: {}, type: {}".format(prediction[0], type(prediction[0])))
            else:
                two_dim_array = np.array([form_params[SklearnRESTfulServing.JSON_KEY_NAME]])
                prediction = self._classifier.predict(two_dim_array)

            return (200, {"prediction": prediction[0]})


if __name__ == '__main__':
    import argparse

    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to listen on")
    parser.add_argument("input_model", help="Path of input model to create")
    parser.add_argument("--log_level", choices=log_levels.keys(), default="info", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)-15s %(levelname)s [%(module)s:%(lineno)d]:  %(message)s')
    logging.getLogger('parallelm').setLevel(log_levels[args.log_level])

    SklearnRESTfulServing.run(port=args.port, model_path=args.input_model)
