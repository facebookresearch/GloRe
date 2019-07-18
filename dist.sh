# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

HOST_FILE=./Hosts
WORLD_SIZE=$(cat ${HOST_FILE} | wc -l)


cat ./Hosts 2>&1 | tee -a .full-record.log

python train_kinetics.py \
--world-size ${WORLD_SIZE} \
--lr-base 0.04 \
2>&1 | tee -a .full-record.log

# --resume-epoch 20
