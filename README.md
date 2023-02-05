# kaggle-otto2

## Setup

```bash
./bin/setup.sh
```

## Run

```bash
# DEV (CV with sessions 1/20 sampled)
./bin/run.sh exp001_dev

# CV
./bin/run.sh exp001_cv

# For Submission
./bin/run.sh exp001
```

PYTHONPATH=. python kaggle_otto2/data_loader/main.py --exp exp001_dev
