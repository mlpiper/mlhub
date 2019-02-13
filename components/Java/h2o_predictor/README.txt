#!/bin/bash

# --- copy mlcomp jar, and (h2o generated MOJO jar file) h2o-genmodel jar ----
# a) copy over  ../../../../../../reflex-common/mlcomp/target/mlcomp.jar .
# b) copy over <h2o's MOJO>h2o-genmodel.jar
# c) provide the model to be used for inference along with data-set for inference

java -cp ./mlcomp.jar:./target/h2o_predictor/h2o_predictor.jar:h2o-genmodel.jar \
	com.parallelm.components.h2o_predictor.H2OPredictor \
        --input-model ./GBM_model_python_1545102497965_1.zip \
        --samples-file ./prostate-med-test.csv \
	--output-file ./Results.csv
