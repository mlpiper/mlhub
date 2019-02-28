#!/usr/bin/env bash

curl -s -G http://localhost:8888/predict -d '{"AGE":68,"RACE":2,"DCAPS":2,"VOL":0,"GLEASON":6}'

