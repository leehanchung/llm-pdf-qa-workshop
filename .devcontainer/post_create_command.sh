#!/bin/bash
set -euf -o pipefail
IFS=$'\n\t'

# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create python virtual environment
poetry install --no-root
poetry shell
