#!/bin/bash

#SBATCH --job-name=yo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=2
#SBATCH --time=0-22:00:00
##SBATCH --mem=16G
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=dgx
#SBATCH --gres=gpu:teslav100:2
#SBATCH --output=../logs/%x-%j-yolov8_air.out
#SBATCH --error=../logs/yolov8_air.err

# Activate conda environment

if [ -d /data/VirtualAging ] ;
then
  dVA="VirtualAging"
else
  dVA="virtualaging"
fi

CONDA_ENV_NAME=/data/$dVA/users/ghoyer/conda/envs/autolabel

# Specify the configuration name here
CONFIG_NAME="OAI_Thigh_yolo"

# Activate Conda environment
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}

# Print node in the terminal
echo "Running on node $HOSTNAME"

## Navigate to root directory
cd .. || exit

# Pass the configuration name (without the .yaml extension) to the Python script
echo "Starting training with configuration: ${CONFIG_NAME}"
python src/obj_detection/predict_msk.py "${CONFIG_NAME}"

## End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"  
                                          


# # run as sbatch predict_yolo.sh