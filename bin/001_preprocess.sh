#!/bin/bash

set -exuo pipefail

EXP=$1

PYTHONPATH=. python kaggle_otto2/data_loader/main.py --exp "$EXP"

