#!/bin/bash

set -e

pip install --upgrade -r prereqs.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install --no-build-isolation -r requirements.txt
pip install -r image.txt
