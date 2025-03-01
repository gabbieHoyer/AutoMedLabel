#!/bin/bash

###############################################################################
# SLURM Job Configuration (Single-GPU Evaluation Template)
###############################################################################
#SBATCH --job-name=finetune_eval        # Job name, can be changed to 'eval_only' or other
#SBATCH --output=../logs/%x-%j.out      # Log file (%x = job name, %j = job ID)
#SBATCH --error=../logs/%x-%j.err       # Error file
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --time=24:00:00                 # Wall time limit (hh:mm:ss)
#SBATCH --cpus-per-task=4               # CPU cores per task
#SBATCH --gres=gpu:1                    # Number of GPUs
#SBATCH --mem=32G                       # Memory per node
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
# You can either:
#    (1) Hardcode the subcommand and config path, or 
#    (2) Pass them as arguments to this script.

# Example (1) Hardcode them:
SUBCOMMAND="eval"  # or "eval_biomarker", "eval_det2seg", etc.
CONFIG_FILE="<CONFIG_FILE_PATH_OR_BASENAME>"  # e.g., "configs/my_eval_experiment.yaml"

# Example (2) If you want to parse arguments from the command line:
# SUBCOMMAND="$1"
# CONFIG_FILE="$2"
# [ -z "$SUBCOMMAND" ] && echo "Error: No subcommand specified." && exit 1
# [ -z "$CONFIG_FILE" ] && echo "Error: No config file specified." && exit 1

###############################################################################
# 3. Print Diagnostic Information
###############################################################################
echo "Running on node: $HOSTNAME"
echo "Using subcommand: ${SUBCOMMAND}"
echo "Using config file: ${CONFIG_FILE}"

# Optionally move to project root:
# cd /path/to/your/project/root || exit

###############################################################################
# 4. Run the Subcommand via src.main
###############################################################################
# The subcommands can be:
#   - eval
#   - eval_biomarker
#   - eval_det2seg
#   - (etc. as defined in src.main)
echo "Starting the ${SUBCOMMAND} process..."
python -m src.main "${SUBCOMMAND}" "${CONFIG_FILE}"
echo "${SUBCOMMAND} completed successfully for config: ${CONFIG_FILE}"

###############################################################################
# 5. (Optional) Post-Job Diagnostics
###############################################################################
if [ -n "$JOB_ID" ]; then
  qstat -j "$JOB_ID"
fi

echo "All processes completed successfully."
