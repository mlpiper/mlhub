#!/bin/bash
if [ -z $REFLEX_HOME ];
then PYTHONPATH="../:$HOME/workspace/mlpiper/mlcomp/dist/mlcomp-py2.egg:$HOME/workspace/mlpiper/mlops/dist/mlops-py2.egg";
else 
    if [ -d $REFLEX_HOME ];
    then
       PYTHONPATH="../:$REFLEX_HOME/sub/mlpiper/mlcomp/dist/mlcomp-py2.egg:$REFLEX_HOME/sub/mlpiper/mlops/dist/mlops-py2.egg";
    else
       echo "REFLEX_HOME is invalid $REFLEX_HOME"
       exit 1
    fi
fi

script_name=$(basename ${BASH_SOURCE[0]})
script_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))


PYTHONPATH=$PYTHONPATH python pmml_restful_serving.py 8888 $script_dir/model/modelForRf
