from enum import Enum
import logging
import numpy as np
import os
import pickle
import pprint
import sklearn
import sys
import warnings

from parallelm.components.restful.flask_route import FlaskRoute
from parallelm.components.restful_component import RESTfulComponent
from parallelm.components.restful.metric import Metric, MetricType, MetricRelation


class ModelType(Enum):
    other = 1
    classifier = 2
    regressor = 3
    clusteror = 4


class SklearnRESTfulServing(RESTfulComponent):
    JSON_KEY_NAME = "data"

    def __init__(self, engine):
        super(SklearnRESTfulServing, self).__init__(engine)
        self._model = None
        self._model_type = ModelType.other
        self._model_loading_error = None
        self._params = {}
        self._verbose = self._logger.isEnabledFor(logging.DEBUG)
        self._num_predictable_classes = 0
        self._confidence_metric_per_class = []
        self._num_predictions_metric_per_class = []

        self._low_confidence_threshold_percent = None

        # Metrics
        self._num_confidences_below_threshold = None
        self._total_confidence_metric = None
        self._confidence_bar_graph_metric = None
        self._prediction_distribution_bar_graph_metric = None

        self.info_json = {
            "sample_keyword": SklearnRESTfulServing.JSON_KEY_NAME,
            "python": "{}.{}.{}".format(sys.version_info[0], sys.version_info[1], sys.version_info[2]),
            "numpy": np.version.version,
            "sklearn": sklearn.__version__,
        }

    def configure(self, params):
        """
        @brief      It is called in within the 'deputy' context
        """
        self._logger.info("Configure component with input params, name: {}, params: {}"
                          .format(self.name(), params))
        self._params = params

        self._total_confidence_metric = Metric("total.confidence",
                                               title="Average Confidence",
                                               metric_type=MetricType.COUNTER_PER_TIME_WINDOW,
                                               value_type=float,
                                               metric_relation=MetricRelation.AVG_PER_REQUEST)

        self._num_predictable_classes = self._params.get("num_predictable_classes", 0)
        if self._num_predictable_classes > 0:

            # Prediction distribution bar graph
            self._prediction_distribution_bar_graph_metric = Metric("prediction_distribution",
                                                                    title="Prediction Distribution",
                                                                    metric_type=MetricType.COUNTER_PER_TIME_WINDOW,
                                                                    metric_relation=MetricRelation.BAR_GRAPH,
                                                                    related_metric=[])
            for iii in range(self._num_predictable_classes):
                metric = Metric("num.prediction.per.class.{}".format(iii),
                                hidden=True,
                                metric_type=MetricType.COUNTER_PER_TIME_WINDOW)
                self._num_predictions_metric_per_class.append(metric)
                self._prediction_distribution_bar_graph_metric.add_related_metric((metric, "{}".format(iii)))

            # Confidence bar graph
            self._confidence_bar_graph_metric = Metric("confidence_bar_graph",
                                                       title="Average Confidence per class",
                                                       metric_type=MetricType.COUNTER_PER_TIME_WINDOW,
                                                       metric_relation=MetricRelation.BAR_GRAPH,
                                                       related_metric=[])

            for iii in range(self._num_predictable_classes):
                metric = Metric("total.confidence.per.class.{}".format(iii),
                                hidden=True,
                                metric_type=MetricType.COUNTER_PER_TIME_WINDOW,
                                value_type=float,
                                metric_relation=MetricRelation.DIVIDE_BY,
                                related_metric=self._num_predictions_metric_per_class[iii])
                self._confidence_metric_per_class.append(metric)
                self._confidence_bar_graph_metric.add_related_metric((metric, "{}".format(iii)))

        self._low_confidence_threshold_percent = self._params.get("low_confidence_threshold_percent", 0)
        if self._low_confidence_threshold_percent > 0:
            self._num_confidences_below_threshold = Metric("num.confidence.below.thrsh",
                                                           title="Number of predictions with confidence below {}% threshold"
                                                           .format(self._low_confidence_threshold_percent),
                                                           metric_type=MetricType.COUNTER_PER_TIME_WINDOW)

    def load_model_callback(self, model_path, stream, version):
        self._logger.info(sys.version_info)

        self._logger.info("Model is loading, wid: {}, path: {}".format(self.get_wid(), model_path))
        self._logger.info("params: {}".format(pprint.pformat(self._params)))
        model = None

        with warnings.catch_warnings(record=True) as warns:
            try:
                with open(model_path, "rb") as f:
                    self._model_loading_error = None
                    model = pickle.load(f) if sys.version_info[0] < 3 \
                        else pickle.load(f, encoding='latin1')

                    if self._verbose:
                        self._logger.debug("Un-pickled model: {}".format(self._model))
                    self._logger.debug("Model loaded successfully!")

            except Exception as e:
                warn_str = ""
                if len(warns) > 0:
                    warn_str = "{}".format(warns[-1].message)
                self._logger.error("Model loading warning: {}; Model loading error: {}".format(warn_str, e))

                # Not sure we want to throw exception only to move to a non model mode
                if self._params.get("ignore-incompatible-model", True):
                    self._logger.info("New model could not be loaded, due to error: {}".format(e))
                    if self._model is None:
                        self._model_loading_error = "Model loading warning: {}; Model loading error: {}".format(warn_str, str(e))
                    else:
                        raise Exception("Model loading warning: {}; Model loading error: {}".format(warn_str, e))

        # This line should be reached only if
        #  a) model loaded successfully
        #  b) model loading failed but it can be ignored
        if model is not None:
            self._model = model
            self._update_model_type()

    def _update_model_type(self):
        if self._model:
            if sklearn.base.is_classifier(self._model) or getattr(self._model, "_estimator_type", None) == "clusterer":
                self._model_type = ModelType.classifier
            elif sklearn.base.is_regressor(self._model):
                self._model_type = ModelType.regressor
            else:
                self._model_type = ModelType.other

    def _empty_predict(self):
        model_loaded = True if self._model else False

        result_json = {
            "message": "got empty predict",
            "expected_input_format" : "{{\"data\":[<vector>]}}",
            "model_loaded": model_loaded,
            "model_class": str(type(self._model))
        }

        if model_loaded is False and self._model_loading_error:
            result_json["model_load_error"] = self._model_loading_error

        if self._model:
            if hasattr(self._model, "n_features_"):
                result_json["n_features"] = self._model.n_features_
                result_json["expected_input_format"] += ", where vector has {} comma separated values".format(self._model.n_features_)

        result_json.update(self.info_json)

        return result_json

    @FlaskRoute('/predict')
    def predict(self, url_params, form_params):

        if len(form_params) == 0:
            return 200, self._empty_predict()

        elif not self._model:
            if self._model_loading_error:
                return_json = {"error": "Failed loading model: {}".format(self._model_loading_error)}
            else:
                return_json = {"error": "Model not loaded yet - please set a model"}
            return_json.update(self.info_json)
            return 404, return_json

        elif SklearnRESTfulServing.JSON_KEY_NAME not in form_params:
            msg = "Unexpected json format for prediction! Missing '{}' key in: {}" \
                .format(SklearnRESTfulServing.JSON_KEY_NAME, form_params)
            self._logger.error(msg)
            error_json = {"error": msg}
            error_json.update(self.info_json)
            return 404, error_json
        else:
            try:
                two_dim_array = np.array([form_params[SklearnRESTfulServing.JSON_KEY_NAME]])
                pred_probs = None
                try:
                    pred_probs = self._model.predict_proba(two_dim_array)[0]
                except:
                    prediction = self._model.predict(two_dim_array)[0]

                if pred_probs is not None:
                    pred_index = np.argmax(pred_probs)
                    prediction = self._model.classes_[pred_index]
                    prediction_confidence = pred_probs[pred_index]

                    self._logger.debug("pred_probs: {}, pred_index: {}, prediction: {}, confidence: {}"
                                       .format(pred_probs,  pred_index, prediction, prediction_confidence))

                    # Total confidence
                    self._total_confidence_metric.increase(prediction_confidence)

                    if self._num_predictable_classes:
                        # Prediction confidence per class
                        # index = int(prediction * Metric.FLOAT_PRECISION) % self._num_predictable_classes
                        index = int(prediction)
                        self._confidence_metric_per_class[index].increase(prediction_confidence)

                    # Lower probability threshold
                    if self._low_confidence_threshold_percent and \
                            prediction_confidence * 100 < self._low_confidence_threshold_percent:
                        self._num_confidences_below_threshold.increase(1)

                if self._model_type == ModelType.classifier and self._num_predictable_classes:
                    # index = int(prediction * Metric.FLOAT_PRECISION) % self._num_predictable_classes
                    index = int(prediction)
                    self._num_predictions_metric_per_class[index].increase(1)

                if self._verbose:
                    self._logger.debug("predict, url_params: {}, form_params: {}".format(url_params, form_params))
                    self._logger.debug("type<form_params>: {}\n{}".format(type(form_params), form_params))
                    self._logger.debug("type(two_dim_array): {}\n{}".format(type(two_dim_array), two_dim_array))
                    self._logger.debug("prediction: {}, type: {}".format(prediction, type(prediction)))
                return 200, {"prediction": prediction}
            except Exception as e:
                error_json = {"error": "Error performing prediction: {}".format(e)}
                error_json.update(self.info_json)
                return 404, error_json


if __name__ == '__main__':
    import argparse

    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to listen on")
    parser.add_argument("input_model", help="Path of input model to create")
    parser.add_argument("--log_level", choices=log_levels.keys(), default="info", help="Logging level")
    args = parser.parse_args()

    if not os.path.exists(args.input_model):
        raise Exception("Model file {} does not exists".format(args.input_model))

    logging.basicConfig(format='%(asctime)-15s %(levelname)s [%(module)s:%(lineno)d]:  %(message)s')
    logging.getLogger('parallelm').setLevel(log_levels[args.log_level])

    SklearnRESTfulServing.run(port=args.port, model_path=args.input_model)
