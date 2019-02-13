#!/bin/bash

script_name=$(basename ${BASH_SOURCE[0]})
script_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))

PYTHONPATH="../:$HOME/dev/reflex/sub/reflex-algos/target/mlcomp-1.0-py2.7.egg:$HOME/dev/reflex/sub/reflex-algos/target/parallelm-1.0.1-py2.7.egg"

PYTHONPATH=$PYTHONPATH python h2o_restful_serving.py 8888 $script_dir/model/GBM_model_python_1542317998658_5.zip
