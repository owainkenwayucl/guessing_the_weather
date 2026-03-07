#!/bin/bash

set -e

# On GB10 we have memory issues building Flash Attention
export MAX_JOBS=4
export NVCC_THREADS=2
export CMAKE_BUILD_PARALLEL_LEVEL=4
export FLASH_ATTENTION_FORCE_BUILD="TRUE"

# Download wheel from ARC wheel shop
#wget https://wheelshop.arc-general.condenser.arc.ucl.ac.uk/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl
#wget https://www.ucl.ac.uk/~uccaoke/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl
cp ~/shared/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl .

sha256sum -c flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl.sha256

pip install --upgrade -r prereqs.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install ./flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl
pip install --no-build-isolation -r requirements.txt
pip install -r image.txt
