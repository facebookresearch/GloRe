#!/usr/bin/env bash

# Mainly for debugging

python train_kinetics.py \
--batch-size 64
# --lr-base 0.02 \
# --gpus "0,1" \
# 2>&1 | tee -a .full-record.log
