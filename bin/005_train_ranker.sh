#!/bin/bash

set -exuo pipefail

EXP=$1

PYTHONPATH=. python kaggle_otto2/ranker/main.py --exp "$EXP"
