#!/bin/bash

EXP=$1
MESSAGE=$2
CV_SCORE=$3

kaggle competitions submit \
    -c otto-recommender-system \
    -f output/$EXP/submission.csv \
    -m "$EXP(CV$CV_SCORE): $MESSAGE"
