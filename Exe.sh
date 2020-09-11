#!/bin/bash
BASE_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
BASE_DIR="$( realpath "${BASE_DIR}/..")"

export CUDA_DIR=/usr/local/cuda-10.1
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


SIZE=256
batch=8
lr=1e-4
FOLDER="U2/0910"
python3 Train.py \
    -g GenJson \
    -m U2Net \
    -b ${batch}  \
    -e 200 -s ${SIZE} \
    -l $lr \
    -v 1 \
    -t2 /mnt/e/Glaucoma/Seg_Code/Json/Test0910.json \
    -t /mnt/e/Glaucoma/Seg_Code/Json/Val0910.json \
    -T /mnt/e/Glaucoma/Seg_Code/Json/Train0910.json \
    -I 100 \
    -D $FOLDER \
    -mp 0 -xla 0 \
    -O  "tfa.optimizers.AdamW(lr=${lr}, weight_decay = 1e-6)"





