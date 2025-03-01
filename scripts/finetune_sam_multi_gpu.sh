#!/bin/bash

###############################################################################
# SLURM Job Configuration (Multi-GPU Generic Template)
###############################################################################
#SBATCH --job-name=finetune            # Job name
#SBATCH --output=../logs/finetune_%j.log   # Output log file (%j = job ID)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --time=30:00:00                # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:4                   # Number of GPUs per node (e.g., 4)
#SBATCH --cpus-per-task=8              # CPU cores per task
#SBATCH --mem=128G                     # Memory requirement
# Optional directives:
# #SBATCH --partition=<PARTITION_NAME>
# #SBATCH --nodelist=<NODE_NAME>
# #SBATCH --ntasks-per-node=4

# Exit immediately on error
set -e

echo "Running on node: $HOSTNAME"
echo "Current directory: $(pwd)"

###############################################################################
# 1. Environment Setup
###############################################################################
# If you require modules on your HPC, load them here (placeholder):
# module load anaconda/3
# module load cuda/11.0

# Source a script that configures distributed training environment variables.
# This script presumably sets MASTER_ADDR, MASTER_PORT, GPUS_PER_NODE, etc.
# e.g., 'env_scripts/set_env.sh' calls 'env_scripts/distributed_training_env.sh'
# and then does `source distributed_training_env.sh`.
chmod +x env_scripts/set_env.sh
source env_scripts/set_env.sh

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "DIST_URL=$DIST_URL"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "NNODES=$NNODES"
echo "RANK=$RANK"

###############################################################################
# 2. Conda Environment Activation (Placeholder)
###############################################################################
# Adjust <CONDA_ENV_PATH> or <CONDA_ENV_NAME> to match your HPC environment
CONDA_ENV_NAME="<CONDA_ENV_PATH_OR_NAME>"  # e.g., "/data/virtualaging/users/username/conda/envs/autolabel"

echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source activate "${CONDA_ENV_NAME}" || conda activate "${CONDA_ENV_NAME}"

###############################################################################
# 3. User-Defined Parameters
###############################################################################
# Replace this with your actual config file (without or with .yaml extension)
CONFIG_NAME="OAI_T1_Thigh"

echo "Config chosen for finetuning pipeline: ${CONFIG_NAME}"

###############################################################################
# 4. Navigate to Project Root (Optional)
###############################################################################
cd .. || exit  # Adjust if your script already resides at project root

###############################################################################
# 5. Run Distributed Finetuning with Torchrun
###############################################################################
echo "Starting distributed training using torchrun..."
torchrun --nproc_per_node="${GPUS_PER_NODE}" \
         --nnodes="${NNODES}" \
         --node_rank="${RANK}" \
         --master_addr="${MASTER_ADDR}" \
         --master_port="${MASTER_PORT}" \
         -m src.main finetune "${CONFIG_NAME}"

# Note:
# The '-m' argument runs the module 'src.main' as a script.
# 'finetune' is the subcommand, and CONFIG_NAME is the config argument.

echo "Finetuning completed successfully for config: ${CONFIG_NAME}"

###############################################################################
# 6. (Optional) Post-Job Diagnostics
###############################################################################
if [ -n "$JOB_ID" ]; then
  qstat -j "$JOB_ID"
fi

echo "All processes completed successfully."
