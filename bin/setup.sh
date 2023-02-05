#!/bin/bash

set -exuo pipefail

# Download datasets
./bin/download.sh

# Install packages
poetry install
