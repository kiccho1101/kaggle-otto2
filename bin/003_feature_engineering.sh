#!/bin/bash

set -exuo pipefail

EXP=$1

PYTHONPATH=. python kaggle_otto2/feature/main.py --exp "$EXP"
