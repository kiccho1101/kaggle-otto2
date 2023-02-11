#!/bin/bash

set -exuo pipefail

EXP=$1
MODEL_TYPE=$2

PYTHONPATH=. python kaggle_otto2/ranker_trainer/main.py --exp "$EXP" --model_type "$MODEL_TYPE"
