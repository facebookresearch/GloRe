# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env bash

# Mainly for debugging

python train_kinetics.py \
--batch-size 64
# --lr-base 0.02 \
# --gpus "0,1" \
# 2>&1 | tee -a .full-record.log
