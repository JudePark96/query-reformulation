#!/bin/bash

export PYTHONPATH=../../src

CORES=$(lscpu | grep Core | awk '{print $4}')
SOCKETS=$(lscpu | grep Socket | awk '{print $2}')
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

poetry run accelerate launch --num_processes=3 --mixed_precision=fp16 src/eval_utils/generate_embeddings.py \
                             --config_file ./configs/search/topiOCQA_generate_embedding.yaml
