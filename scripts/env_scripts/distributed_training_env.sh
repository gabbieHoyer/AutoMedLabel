#!/bin/bash
export MASTER_ADDR=rhea
export MASTER_PORT=27565
export GPUS_PER_NODE=4
export NNODES=1
export WORLD_SIZE=4
export RANK=0
export LOCAL_RANK=0
export DIST_BACKEND=nccl
export DIST_URL=tcp://rhea:27565
