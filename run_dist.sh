#!/usr/bin/env bash

PWD_PATH=$(cd ./ && pwd)

HOST_FILE=./Hosts

SESSION_NAME='run-dist'
CMD_NEW_SESSION="tmux new -s ${SESSION_NAME} -d | true"
CMD_RUN='export PATH=/private/home/yunpeng/opt/anaconda3/envs/pytorch-4.1/bin:/private/home/yunpeng/opt/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin; export LD_LIBRARY_PATH=/private/home/yunpeng/local/cuda-9.0/lib64:/private/home/yunpeng/opt/cudnn/v7.2.1.38-4-9.0/lib64:/private/home/yunpeng/opt/nccl/v2.2.13-4-9.0/lib:/private/home/yunpeng/opt/anaconda3/envs/pytorch-4.1/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib64; '
CMD_RUN+="cd ${PWD_PATH}; "
CMD_RUN+="bash dist.sh; "

read -p "-> Press any key to start distributed training."

cat ${HOST_FILE} | xargs -I{} ssh {} "hostname && ${CMD_NEW_SESSION} && tmux send -t \"${SESSION_NAME}\" \"${CMD_RUN}\" ENTER"
