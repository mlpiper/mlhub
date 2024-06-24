#!/bin/bash

MODEL=$1
DATA_FILE=$2

java -cp ./target/pmml_predictor/pmml_predictor.jar \
    org.mlpiper.mlhub.components.pmml_predictor.PmmlPredictor \
    --convert-invalid-numbers-to-na true \
    --convert-unknown-categorical-levels-to-na true \
    --input-model $MODEL \
    --samples-file $DATA_FILE \
    --output-file /tmp/predictions.csv

