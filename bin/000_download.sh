#!/bin/bash

set -exo pipefail

download_competition() {
    DIR_NAME=$1

    mkdir -p input

    # Download competition data
    kaggle competitions download -c "$DIR_NAME"

    # Unzip
    unzip "$DIR_NAME.zip" -d "input/$DIR_NAME"

    # Remove zip
    rm "$DIR_NAME.zip"
}

download_dataset() {
    DATASET_NAME=$1
    DIR_NAME=$2

    mkdir -p input

    # Download dataset data
    kaggle datasets download -d "$DATASET_NAME"

    # Unzip
    unzip "$DIR_NAME.zip" -d "input/$DIR_NAME"

    # Remove zip
    rm "$DIR_NAME.zip"
}

download_competition otto-recommender-system
download_dataset "radek1/otto-full-optimized-memory-footprint" "otto-full-optimized-memory-footprint"
download_dataset "radek1/otto-train-and-test-data-for-local-validation" "otto-train-and-test-data-for-local-validation"
