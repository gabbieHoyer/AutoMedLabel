#!/bin/bash

# Name of the file to which we will write the environment variables
# ENV_FILE="distributed_training_env.sh"

# Get the directory of the current script to ensure paths are relative to this script's location
SCRIPT_DIR="$(dirname "$BASH_SOURCE")"
ENV_FILE="$SCRIPT_DIR/distributed_training_env.sh"

# Assume the MASTER_ADDR is the hostname of the first node allocated by SLURM
MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)

MASTER_PORT=$((($SLURM_JOB_ID % 10000) + 20000)) # Example to get a port in the range 20000-29999

# Other necessary variables
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
NNODES=${SLURM_NNODES:-1}
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
RANK=${SLURM_PROCID:-0}
LOCAL_RANK=${SLURM_LOCALID:-0}
DIST_BACKEND='nccl'
DIST_URL="tcp://${MASTER_ADDR}:${MASTER_PORT}"

# Write to file
echo "#!/bin/bash" > $ENV_FILE
echo "export MASTER_ADDR=${MASTER_ADDR}" >> $ENV_FILE
echo "export MASTER_PORT=${MASTER_PORT}" >> $ENV_FILE
echo "export GPUS_PER_NODE=${GPUS_PER_NODE}" >> $ENV_FILE
echo "export NNODES=${NNODES}" >> $ENV_FILE
echo "export WORLD_SIZE=${WORLD_SIZE}" >> $ENV_FILE
echo "export RANK=${RANK}" >> $ENV_FILE
echo "export LOCAL_RANK=${LOCAL_RANK}" >> $ENV_FILE
echo "export DIST_BACKEND=${DIST_BACKEND}" >> $ENV_FILE
echo "export DIST_URL=${DIST_URL}" >> $ENV_FILE

# check to make sure this still works since creating subdirectory within scripts for clarity

echo "The current directory is:"
pwd

echo "node env variable is $ENV_FILE"

echo "chmod +x $ENV_FILE"

# Make the environment file executable
chmod +x $ENV_FILE

# Source the environment file
source $ENV_FILE






# Dynamically find a free port
# get_free_port() {
#     while true; do
#         PORT=$(shuf -i 20000-25000 -n 1)  # Random port in the range.
#         ss -tulwn | grep -q ":$PORT " || echo $PORT && break  # Check if the port is available.
#     done
# }

# MASTER_PORT=$(get_free_port)



# Use SLURM_JOB_ID to get a unique port, or set a fixed port if that's preferred
# MASTER_PORT=$((($SLURM_JOB_ID % 10000) + 20000)) # Example to get a port in the range 20000-29999
# MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4) + $SLURM_ARRAY_TASK_ID)

# get_free_port() {
#     while true; do
#         PORT=$(shuf -i 20000-25000 -n 1)  # Random port in the range.
#         ss -tulwn | grep -q ":$PORT " || echo $PORT && break  # Check if the port is available.
#     done
# }

# MASTER_PORT=$(get_free_port)

# MASTER_PORT=$(python -c 'import socket; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.bind(("",0)); print(sock.getsockname()[1]); sock.close()')
