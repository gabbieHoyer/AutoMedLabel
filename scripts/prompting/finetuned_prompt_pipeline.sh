#!/bin/bash

#SBATCH --job-name=infer_dataset
#SBATCH --output=../logs/fine_infer_dataset_%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=24G
##SBATCH --partition=dgx
#SBATCH --partition=gpu
##SBATCH --gres=gpu:teslav100:1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

echo "Running on node $HOSTNAME"

# Activate conda environment
export PATH=/netopt/rhel7/bin:$PATH
eval "$('/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"

# Your Conda environment name: 'myenv'
# CONDA_ENV_NAME=/data/VirtualAging/users/ghoyer/conda/envs/medsam

# ----------------- CONFIG OPTIONS -----------------
# Knee
CONFIG_LIST=("Knee", "TBrecon", "P50_AFACL") 
# Spine
CONFIG_LIST=("Spine") 
# Other anatomy
CONFIG_LIST=("OAI_T1_Thigh")
# Braod Experiments
CONFIG_LIST=("MSK", "Non_Clinical")

# --------------- USER DEFINED PARAMS ----------------
CONFIG_LIST=("TBrecon")

# -------------------- BEGIN CODE --------------------

## Activate Conda environment
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}

## Navigate to the base github directory
cd .. || exit
echo "The current directory is:"
pwd

## Iterate through each dataset
for CONFIG_NAME in "${CONFIG_LIST[@]}"; do 

  echo "**************************************************"
  echo "config $CONFIG_NAME chosen for prompt prediction pipeline"

  # Full path to the configuration file, adjusted for your directory structure
  CONFIG_FILE="config/finetuning/experiments/${CONFIG_NAME}.yaml"

  echo "Starting bbox_prompt_viz.py with ${CONFIG_NAME}"
  python3 src/evaluation/visualizing/bbox_prompt_viz.py "${CONFIG_NAME}" --prompt finetuned

  echo "Pausing to allow cleanup process to complete..."
  sleep 60  # Pauses for "x" seconds to allow for cleanup.

  echo "Starting prompt_prediction.py with ${CONFIG_NAME}"
  python3 src/prompting/prompt_prediction.py "${CONFIG_NAME}" --prompt finetuned
  
  echo "Starting gt_pred_viz.py with ${CONFIG_NAME}"
  python3 src/evaluation/visualizing/gt_pred_viz.py "${CONFIG_NAME}" --prompt finetuned

  echo "Pausing to allow cleanup process to complete..."
  sleep 60  # Pauses for "x" seconds to allow for cleanup.

  echo "Starting metrics.py with ${CONFIG_NAME}"
  python3 src/evaluation/results_synthesis/evaluate.py "${CONFIG_NAME}" --prompt finetuned

done 

echo "All processes completed successfully."

# execute from the root/scripts dir as sbatch prompt_pipeline.sh