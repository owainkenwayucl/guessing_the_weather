#!/bin/bash

set -e

pip install --upgrade -r prereqs.txt
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4

# we appear to need to build flash attention from source??
pip install triton==3.5.1
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install
cd ..

# Image libraries
pip install -r requirements.txt
pip install -r image.txt
