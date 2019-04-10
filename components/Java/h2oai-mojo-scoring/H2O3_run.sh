#!/bin/bash

MODEL=$1
DATA_FILE=$2

java -cp build/libs/h2oai-dai-parallelm-scorer.jar \
    ai.h2o.mojo.parallelm.components.H2O3Predictor \
    --convert-invalid-numbers-to-na true \
    --convert-unknown-categorical-levels-to-na true \
    --input-model $MODEL \
    --samples-file $DATA_FILE \
    --output-file /tmp/predict_results_mojo.csv

