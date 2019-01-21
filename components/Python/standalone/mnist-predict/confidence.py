# Copyright 2018 ParallelM, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

### MLOPS start
from parallelm.mlops import mlops
from parallelm.mlops.mlops_mode import MLOpsMode
from parallelm.mlops.stats.bar_graph import BarGraph
### MLOPS end

class ConfidenceTracker(object):
    def __init__(self, track_conf, conf_thresh, conf_percent, output):
        self._track_conf = track_conf
        self._conf_thresh = conf_thresh
        self._conf_percent = conf_percent
        self._output_low_confidence_predictions = output
        self._low_confidence_predictions = 0

        print("track_conf: {}".format(track_conf))
        if track_conf > 0:
            print("conf_thresh: {}".format(conf_thresh))
            print("conf_percent: {}".format(conf_percent))

        categories = ["10", "20", "30", "40", "50", "60", "70", "80", "90","100"]
        self._conf_hist = []
        for i in range(0, 10):
            self._conf_hist.append(0)

        ## MLOps start
        self._conf_graph = BarGraph().name("Confidence Distribution Bar Graph").cols(categories)
        ## MLOps end

    def check_confidence(self, confidence, sample):

        if self._track_conf == 0:
            return

        conf_bin = int(math.floor(confidence / 10))

        # include 100% confidence in the 90-100 range
        if conf_bin == 10:
            conf_bin = 9

        self._conf_hist[conf_bin] += 1

        if confidence < self._conf_thresh:
	    	self._low_confidence_predictions += 1

	        if self._output_low_confidence_predictions != 0:
	            import tensorflow as tf
	            import matplotlib
	            matplotlib.use('Agg')
	            import matplotlib.pyplot as plt
	            image = tf.reshape(sample,[28,28])
	            plotData = sample
	            plotData = plotData.reshape(28, 28)
	            plt.gray() # use this line if you don't want to see it in color
	            plt.imshow(plotData)
	            plt.savefig("/opt/data-lake/image{}_conf{}_prediction{}.png".format(total_predictions, int(round(confidence)),
	                prediction))

    def report_confidence(self, total_predictions):

        if self._track_conf == 0:
            return

        ## MLOps start
        # Show the prediction distribution as a bar graph
        self._conf_graph.data(self._conf_hist)
        mlops.set_stat(self._conf_graph)
        ## MLOps end

		# Percentage of low confidence predictions in this reporting interval
        low_conf_percent = self._low_confidence_predictions * 100.0 / total_predictions

        print("low confidence predictions: {} ({})%".format(self._low_confidence_predictions, low_conf_percent))

        if low_conf_percent > self._conf_percent:
            msg = "Low confidence: {}% of inferences had confidence below {}%".format(low_conf_percent, self._conf_thresh)
            print(msg)

        	## MLOps start
            mlops.health_alert("Low confidence alert", msg)
            ## MLOps end

        # reset counters for next round
        for i in range(0, 9):
            self._conf_hist[i] = 0
        self._low_confidence_predictions = 0