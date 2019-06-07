#!/bin/bash

# --- copy mlcomp jar, and (jpmml generated MOJO jar file) ----
# a) copy over  ../../../../../../reflex-common/mlcomp/target/mlcomp.jar .
# b) provide the model to be used for inference along with data-set for inference

java -cp ./mlcomp.jar:./target/pmml_predictor/pmml_predictor.jar \
    org.mlpiper.mlhub.components.pmml_predictor.PmmlPredictor \
        --input-model <model_file> \
        --samples-file <data_file> \
	    --output-file ./Results.csv
