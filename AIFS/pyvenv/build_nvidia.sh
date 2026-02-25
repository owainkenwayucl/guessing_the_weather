#!/bin/bash

set -e

# On GB10 we have memory issues building Flash Attention
export MAX_JOBS=4
export NVCC_THREADS=2
export CMAKE_BUILD_PARALLEL_LEVEL=4
export FLASH_ATTENTION_FORCE_BUILD="TRUE"

pip install --upgrade -r prereqs.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install --no-build-isolation -r requirements.txt
pip install -r image.txt
