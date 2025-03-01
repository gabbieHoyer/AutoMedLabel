#!/bin/bash

###############################################################################
# SLURM Job Configuration (Single-GPU Finetune Template)
###############################################################################
#SBATCH --job-name=finetune_training         # Job name
#SBATCH --output=../logs/%x-%j.out           # Log file (%x = job name, %j = job ID)
#SBATCH --error=../logs/%x-%j.err            # Error file
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --time=24:00:00                      # Wall time limit (hh:mm:ss)
#SBATCH --cpus-per-task=4                    # CPU cores per task
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH --mem=32G                            # Memory per node
# Add or adjust additional SLURM directives as needed:
#   #SBATCH --partition=<PARTITION_NAME>
#   #SBATCH --nodelist=<SPECIFIC_NODE>
#   #SBATCH --ntasks-per-node=<NUM_TASKS>

# Exit immediately on error
set -e

###############################################################################
# 1. Environment Setup (Placeholder)
###############################################################################
# If your HPC requires modules, load them here, e.g.:
#   module load anaconda/3
#   module load cuda/11.0

# Replace <CONDA_ENV_PATH_OR_NAME> with the actual environment name or path.
CONDA_ENV_NAME=<CONDA_ENV_PATH_OR_NAME>  # e.g., "/data/virtualaging/users/username/conda/envs/autolabel"

echo "Activating conda environment: ${CONDA_ENV_NAME}"
source activate "${CONDA_ENV_NAME}" || conda activate "${CONDA_ENV_NAME}"

###############################################################################
# 2. User-Defined Parameters
###############################################################################
# Replace <CONFIG_FILE_PATH> with the path or base name of your YAML config.
# For example: "configs/my_finetune_experiment.yaml"
CONFIG_FILE_PATH=<CONFIG_FILE_PATH_OR_BASENAME>

###############################################################################
# 3. Print Diagnostic Information
###############################################################################
echo "Running on node: $HOSTNAME"
echo "Using config file: ${CONFIG_FILE_PATH}"

# Optionally move to project root:
# cd /path/to/your/project/root || exit

###############################################################################
# 4. Run Finetune Subcommand (via src.main)
###############################################################################
echo "Starting finetuning..."
python -m src.main finetune "${CONFIG_FILE_PATH}"
echo "Finetuning completed successfully for config: ${CONFIG_FILE_PATH}"

###############################################################################
# 5. (Optional) Post-Job Diagnostics
###############################################################################
if [ -n "$JOB_ID" ]; then
  qstat -j "$JOB_ID"
fi

echo "All processes completed successfully."
