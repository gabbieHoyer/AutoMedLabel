#!/bin/bash

###############################################################################
# SLURM Job Configuration (Generic Template)
###############################################################################
#SBATCH --job-name=object_detection_training  # Job name
#SBATCH --output=../logs/%x-%j.out            # Log file (%x = job name, %j = job ID)
#SBATCH --error=../logs/%x-%j.err             # Error file
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --time=24:00:00                       # Wall time limit (hh:mm:ss)
#SBATCH --cpus-per-task=4                     # CPU cores per task
#SBATCH --gres=gpu:1                          # Number of GPUs
#SBATCH --mem=32G                             # Memory per node
# Add or adjust additional SLURM directives as needed:
#   #SBATCH --partition=<PARTITION_NAME>
#   #SBATCH --nodelist=<SPECIFIC_NODE>
#   #SBATCH --ntasks-per-node=<NUM_TASKS>

# Exit immediately on error
set -e

###############################################################################
# Environment Setup (Placeholder)
###############################################################################
# 1. Load modules if your HPC requires them. For example:
#    module load anaconda/3
#    module load cuda/11.0

# 2. Activate your conda environment or another environment if needed.
#    Replace <CONDA_ENV_PATH_OR_NAME> with the actual environment name or path.
CONDA_ENV_NAME=<CONDA_ENV_PATH_OR_NAME>  # e.g., "/data/virtualaging/users/username/conda/envs/autolabel"

echo "Activating conda environment: ${CONDA_ENV_NAME}"
source activate "${CONDA_ENV_NAME}" || conda activate "${CONDA_ENV_NAME}"

###############################################################################
# User-Defined Parameters
###############################################################################
# Replace <CONFIG_FILE_PATH> with the path (or base name) to your config file.
# If using the '-m src.main' approach, you can provide the YAML path directly.
CONFIG_FILE_PATH=<CONFIG_FILE_PATH_OR_BASENAME>   # e.g., "configs/my_det_training.yaml"

###############################################################################
# Print Diagnostic Information
###############################################################################
echo "Running on node: $HOSTNAME"
echo "Using config file: ${CONFIG_FILE_PATH}"

# Optionally change directory to the project root if needed:
# cd /path/to/your/project/root || exit

###############################################################################
# Run Object Detection Training (via src.main)
###############################################################################
# Assumes your code has a subcommand "train_det" in src/main.py:
# e.g., python -m src.main train_det <CONFIG_FILE_PATH>

echo "Starting object detection training..."
python -m src.main train_det "${CONFIG_FILE_PATH}"
echo "Training completed successfully for config: ${CONFIG_FILE_PATH}"

###############################################################################
# (Optional) Post-Job Diagnostics
###############################################################################
if [ -n "$JOB_ID" ]; then
  qstat -j "$JOB_ID"
fi

echo "All processes completed successfully."
