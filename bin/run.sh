#!/bin/bash

set -exuo pipefail

EXP=$1

# PYTHONPATH=. poetry run python kaggle_otto2/data_loader/main.py --exp "$EXP"

# PYTHONPATH=. poetry run python kaggle_otto2/cand_generator/last_inter/main.py --exp "$EXP"
PYTHONPATH=. poetry run python kaggle_otto2/cand_generator/item_cf/main.py --exp "$EXP"
