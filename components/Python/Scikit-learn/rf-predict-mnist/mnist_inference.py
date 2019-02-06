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
from __future__ import print_function
import argparse
import numpy as ny
import os.path
import sys
import time
import pickle

from mnist_stream_input import MnistStreamInput

### MLOPS start
from parallelm.mlops import mlops
from parallelm.mlops.mlops_mode import MLOpsMode
from parallelm.mlops.predefined_stats import PredefinedStats
from parallelm.mlops.stats.table import Table
from parallelm.mlops.stats.bar_graph import BarGraph
### MLOPS end

def add_parameters(parser):

    # Input configuration
    parser.add_argument("--randomize_input", dest="random", default=False, action='store_true')
    parser.add_argument("--total_records", type=int, dest="total_records", default=1000, required=False)
    parser.add_argument("--input_dir",
                        dest="input_dir",
                        type=str,
                        required=False,
                        help='Directory for caching input data',
                        default="/tmp/mnist_data")

    # Where to save predictions
    parser.add_argument("--output_file",
                        dest="output_file",
                        type=str,
                        required=False,
                        help='File for output predictions',
                        default="/tmp/mnist_predictions")

    # Model configuration
    parser.add_argument("--model_dir", dest="model_dir", type=str, required=True)

    # Stats arguments
    parser.add_argument("--stats_interval", dest="stats_interval", type=int, default=100, required=False)

    # Alerting configuration
    parser.add_argument("--conf_thresh", dest="conf_thresh", help='Confidence threshold for raising alerts',
                        type=int, default=50, required=False)
    parser.add_argument("--conf_percent", dest="conf_percent", help='Confidence percent for raising alerts',
                        type=int, default=10, required=False)

def infer_loop(model, input, output_file, stats_interval, conf_thresh, conf_percent):

    output = open(output_file, "w")

    # Initialize statistics
    total_predictions = 0
    low_confidence_predictions = 0
    categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    prediction_hist = []
    for i in range(0, len(categories)):
        prediction_hist.append(0)

    ### MLOPS start
    # Create a bar graph and table for reporting prediction distributions and set the column names
    infer_bar = BarGraph().name("Prediction Distribution Bar Graph").cols(categories)
    infer_tbl = Table().name("Prediction Distribution Table").cols(categories)
    ### MLOPS end
    
    while True:
        try:
            sample, label = input.get_next_input()
            sample_np = ny.array(sample).reshape(1, -1)

            
            # The prediction is the class with the highest probability
            prediction = model.predict(sample_np)
            

            # Append the prediction to the output file
            output.write("{}\n".format(prediction))

            # Calculate statistics
            total_predictions += 1
            prediction_hist[ny.int(prediction[0])] += 1

            # Report statistics
            if total_predictions % stats_interval == 0:

                # Report the prediction distribution
                for i in range(0, len(categories)):
                    print("category: {} predictions: {}".format(categories[i], prediction_hist[i]))


                ### MLOPS start


                # Show the prediction distribution as a table
                infer_tbl.add_row(str(total_predictions), prediction_hist)

                # Show the prediction distribution as a bar graph
                infer_bar.data(prediction_hist)

                


        except EOFError:
            # stop when we hit end of input
            # Report the stats
            mlops.set_stat(infer_tbl)
            mlops.set_stat(infer_bar)

            ### MLOPS end
            output.close()

            ### MLOPS start
            mlops.done()
            ### MLOPS end

            break


def main():
    parser = argparse.ArgumentParser()
    add_parameters(parser)
    args = parser.parse_args()

    ### MLOPS start
    # Initialize mlops
    mlops.init()
    ### MLOPS end
    mlops.init()

    if args.model_dir is not None:
        try:
            filename = args.model_dir
            file_obj = open(filename, 'rb')
            mlops.set_stat("model_file", 1)
        except Exception as e:
            print("Model not found")
            print("Got exception: {}".format(e))
            mlops.set_stat("model_file", 0)
            mlops.done()
            return 0

    # load the model
    model = pickle.load(file_obj)

    # get the input
    input = MnistStreamInput(args.input_dir, args.total_records, args.random)
    print('args.total_records:',args.total_records)


    # perform inferences on the input
    infer_loop(model, input, args.output_file, args.stats_interval, args.conf_thresh, args.conf_percent)
    test_data = input._samples
    del model
    del input


if __name__ == "__main__":
    # TF serving client API currently only supports python 2.7
    assert sys.version_info >= (2, 7) and sys.version_info < (2, 8)
    main()
