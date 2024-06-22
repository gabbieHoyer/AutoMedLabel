#!/bin/bash
export MASTER_ADDR=hyperion
export MASTER_PORT=23572
export GPUS_PER_NODE=1
export NNODES=1
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export DIST_BACKEND=nccl
export DIST_URL=tcp://hyperion:23572
