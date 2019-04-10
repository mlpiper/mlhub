#!/bin/bash

MODEL=$1
DATA_FILE=$2
LIC_FILE=$3

java -cp build/libs/h2oai-dai-parallelm-scorer.jar \
    ai.h2o.mojo.parallelm.components.H2ODriverlessAiPredictor \
    --input-model $MODEL \
    --samples-file $DATA_FILE \
    --output-file /tmp/predict_results_mojo.csv \
    --license-file $LIC_FILE


