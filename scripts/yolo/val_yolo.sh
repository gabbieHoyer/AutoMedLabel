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
export PATH=/netopt/rhel7/bin:$PATH
eval "$('/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"

# Your Conda environment name: 'myenv'
CONDA_ENV_NAME=/data/VirtualAging/users/ghoyer/conda/envs/air

# Specify the configuration name here
# CONFIG_NAME="exp1_seg"  

CONFIG_NAME="exp2_obj_det"
# CONFIG_NAME="exp1_obb"
# CONFIG_NAME="exp1_class"

# Activate Conda environment
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}

# Print node in the terminal
echo "Running on node $HOSTNAME"

# Navigate to the script's directory, then to the root directory
cd /data/VirtualAging/users/ghoyer/correcting_rad_workflow/detection/VLS/AirDet


# Run the validation script with the specified configuration
python val_air.py "${CONFIG_NAME}"

## End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID" 
                                          


# # run as sbatch val_yolo.sh