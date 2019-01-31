#!/bin/bash

MODEL=$1
DATA_FILE=$2

java -cp ./target/h2o_predictor/h2o_predictor.jar \
    com.parallelm.components.h2o_predictor.H2OPredictor \
    --convert-invalid-numbers-to-na true \
    --convert-unknown-categorical-levels-to-na true \
    --input-model $MODEL \
    --samples-file $DATA_FILE \
    --output-file /tmp/predictions.csv

