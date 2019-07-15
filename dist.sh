HOST_FILE=./Hosts
WORLD_SIZE=$(cat ${HOST_FILE} | wc -l)


cat ./Hosts 2>&1 | tee -a .full-record.log

python train_kinetics.py \
--world-size ${WORLD_SIZE} \
--lr-base 0.04 \
2>&1 | tee -a .full-record.log

# --resume-epoch 20
