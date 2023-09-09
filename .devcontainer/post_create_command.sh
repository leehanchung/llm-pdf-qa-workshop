#!/bin/bash
set -euf -o pipefail
IFS=$'\n\t'

# install poetry
curl -sSL https://install.python-poetry.org | python3 -

# create python virtual environment
poetry install --no-root
poetry shell
