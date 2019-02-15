#!/usr/bin/env bash

set -e

script_name=$(basename ${BASH_SOURCE[0]})
script_dir=$(realpath $(dirname ${BASH_SOURCE[0]}))
target_component_dir=${script_dir%_*}

cd $script_dir
mvn clean install

rm -rf $target_component_dir
mkdir -p $target_component_dir

touch $target_component_dir/__init__.py
cp $script_dir/component.json $target_component_dir/
cp $script_dir/target/*jar-with-dependencies.jar $target_component_dir/
cp $script_dir/*.py $target_component_dir/

printf "\e[92m\nComponent was generated in:\e[1;95m $target_component_dir\e[0m\n\n"
