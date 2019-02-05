from inference_statistics import InferenceStatistics
import numpy as ny

from parallelm.mlops import mlops
from parallelm.mlops.mlops_mode import MLOpsMode
from parallelm.mlops.predefined_stats import PredefinedStats
from parallelm.mlops.stats.table import Table
from parallelm.mlops.stats.bar_graph import BarGraph

class CategoricalStatistics(InferenceStatistics):
    def __init__(self, print_interval, stats_type, num_categories, conf_thresh, conf_percent, hot_label=True):
        super(CategoricalStatistics, self).__init__(print_interval)
        self._num_categories = num_categories
        self._hot_label = hot_label
        self._stats_type = stats_type
        self._conf_thresh = conf_thresh / 100.0
        self._conf_percent = conf_percent

        # These are useful for development, but should be replaced by mlops library functions
        self._label_hist = []
        self._infer_hist = []
        for i in range(0, self._num_categories):
            self._label_hist.append(0)
            self._infer_hist.append(0)

        if self._stats_type == "python":
            mlops.init(ctx=None, connect_mlops=True, mlops_mode=MLOpsMode.AGENT)
        elif self._stats_type == "file":
            mlops.init(ctx=None, connect_mlops=False, mlops_mode=MLOpsMode.STAND_ALONE)
        else:
            self._stats_type = "none"

        if self._stats_type != "none":
            column_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            self._infer_tbl = Table().name("categories").cols(column_names)
            self._infer_bar = BarGraph().name("categories bar").cols(column_names)

    def infer_stats(self, sample, label, inference):

        # for now, we only process 1 inference at a time
        inference = inference[0]
        prediction = ny.argmax(inference)
        confidence = inference[prediction]
        if confidence < self._conf_thresh:
            self.increment_low_conf()

        self._infer_hist[prediction] += 1

        if label is not None:
            if (self._hot_label):
                label = ny.argmax(label)
            self._label_hist[label] += 1

            if prediction == label:
                self.increment_correct()

        self.increment_total()
        if self.is_time_to_report():
            self.report_stats()

        return prediction

    def report_stats(self):

        # what percentage of the predictions had confidences less than the threshold
        low_conf_percent = self.get_low_conf() * 100.0 / self.get_report_interval()

        if low_conf_percent > self._conf_percent:
            mlops.health_alert("Low confidence alert", "{}% of inferences had confidence below {}%"
                               .format(low_conf_percent, self._conf_thresh * 100))

        for i in range(0, self._num_categories):
            print(i, "label_total =", self._label_hist[i], "infer_total = ", self._infer_hist[i])

        print("total = ", self.get_total(), "total_correct = ",
              self.get_correct())

        category_data = [self._infer_hist[0], self._infer_hist[1], self._infer_hist[2], self._infer_hist[3],
                         self._infer_hist[4], self._infer_hist[5], self._infer_hist[6], self._infer_hist[7],
                         self._infer_hist[8], self._infer_hist[9]]

        self._infer_tbl.add_row(str(self.get_cum_total()), category_data)
        self._infer_bar.data(category_data)

        if self._stats_type != "none":
            mlops.set_stat("correct_percent", self.get_correct() * 100.0 / self.get_total())
            mlops.set_stat(self._infer_tbl)
            mlops.set_stat(self._infer_bar)
            # Update total prediction count with the all new predictions since we last reported.
            mlops.set_stat(PredefinedStats.PREDICTIONS_COUNT, self.get_report_interval())
            print("Completed {} predictions".format(self.get_report_interval()))

        self.reset()

    def __del__(self):
        mlops.done()
        super(CategoricalStatistics, self).__del__()
