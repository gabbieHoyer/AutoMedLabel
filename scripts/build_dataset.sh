#!/bin/bash

###############################################################################
# SLURM Job Configuration
###############################################################################
#SBATCH --job-name=build_dataset         # Job name
#SBATCH --output=../logs/build_dataset_%j.log  # Log file with job ID
#SBATCH --mem=16G                        # Memory requirement
#SBATCH --time=02:00:00                  # Time limit (hh:mm:ss)
#SBATCH --cpus-per-task=4                # Number of CPU cores per task

# Set bash to exit on error
set -e

echo "Running on node $HOSTNAME"

###############################################################################
# HPC Environment Configuration (Generic)
###############################################################################
# 1. (Optionally) Load required modules for your HPC environment.
#    Below are placeholders; adjust as appropriate for your system.
#    For example:
# module load anaconda
# module load python/3.9

# 2. Activate your conda environment or other virtual environment.
#    Replace <ENV_NAME> with your environment’s name or path.
#    Example:
# source activate <ENV_NAME>

# Or if you use mamba or a different approach, place it here.

###############################################################################
# User-Defined Parameters
###############################################################################
# You can store the list of config files in an array for batch processing.
# CONFIG_LIST=("OAI_T1_Thigh" "TBrecon" "YOUR_CONFIG_NAME")

CONFIG_LIST=("OAI_T1_Thigh")

# If your HPC requires a specific working directory, you can cd there:
# cd /path/to/your/project/scripts || exit

echo "========================================================="
echo "Environment setup complete. Starting data processing..."
echo "========================================================="

###############################################################################
# Data Processing Pipeline
###############################################################################
# Example of iterating through the config files and running your scripts.
for CONFIG_NAME in "${CONFIG_LIST[@]}"; do 
    echo "**************************************************"
    echo "Config: $CONFIG_NAME"

    echo "Starting data_standardization.py with ${CONFIG_NAME}"
    python3 -m preprocessing.data_standardization "${CONFIG_NAME}"

    echo "Starting nifti_viz.py with ${CONFIG_NAME}"
    python3 -m utils.visualization.preprocessing.nifti_viz "${CONFIG_NAME}"

    echo "Starting metadata_creation.py (operation A) with ${CONFIG_NAME}"
    python3 -m preprocessing.metadata_creation "${CONFIG_NAME}" --operation A

    echo "Starting metadata_creation.py (operation B) with ${CONFIG_NAME}"
    python3 -m preprocessing.metadata_creation "${CONFIG_NAME}" --operation B

    echo "Starting slice_standardization.py with ${CONFIG_NAME}"
    python3 -m preprocessing.slice_standardization "${CONFIG_NAME}"

    echo "Starting npy_viz.py with ${CONFIG_NAME}"
    python3 -m utils.visualization.preprocessing.npy_viz "${CONFIG_NAME}"

    echo "Starting metadata_creation.py (operation C) with ${CONFIG_NAME}"
    python3 -m preprocessing.metadata_creation "${CONFIG_NAME}" --operation C

    echo "Starting metadata_creation.py (operation D) with ${CONFIG_NAME}"
    python3 -m preprocessing.metadata_creation "${CONFIG_NAME}" --operation D

    echo "All processes for $CONFIG_NAME completed successfully."
done

echo "========================================================="
echo "All dataset processes are finished."
echo "========================================================="

# Usage:
#   sbatch build_dataset.sh
#
# Make sure you have:
#   - Adjusted the SLURM settings (#SBATCH directives) for your HPC needs
#   - Updated environment/module loading
#   - Provided correct CONFIG_LIST entries for your config files
#   - Confirmed the python modules (e.g., python3 -m preprocessing...) match your repo structure
