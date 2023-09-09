#!/bin/bash

# strict mode bash script
set -euo pipefail
IFS=$'\n\t'

# allow git usage
git config --global --add safe.directory "*"

# install poetry
curl -sSL https://install.python-poetry.org | python3 -

# create python virtual environment
poetry install --no-root

# start virtual environment
poetry shell
