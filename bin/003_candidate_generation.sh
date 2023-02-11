#!/bin/bash

set -exuo pipefail

EXP=$1

# PYTHONPATH=. python kaggle_otto2/cand_generator/last_inter/main.py --exp "$EXP"
# PYTHONPATH=. python kaggle_otto2/cand_generator/item_cf/main.py --exp "$EXP"
# PYTHONPATH=. python kaggle_otto2/cand_generator/item_mf/main.py --exp "$EXP"
# PYTHONPATH=. python kaggle_otto2/cand_generator/user_mf/main.py --exp "$EXP"
# PYTHONPATH=. python kaggle_otto2/cand_generator/item2vec/main.py --exp "$EXP"

PYTHONPATH=. python kaggle_otto2/cand_merger/main.py --exp "$EXP"
