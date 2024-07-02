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
# export PATH=/netopt/rhel7/bin:$PATH
# eval "$('/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
export MODULEPATH=$MODULEPATH:/home/ghoyer/Modules/modulefiles
module load use.own

if [ -d "/home/ghoyer/miniconda3" ]; then
    # Load Conda module for RHEL9
    module load conda_base/1.0
    if [ $? -ne 0 ]; then
        echo "Failed to load Miniconda module for RHEL9. Check module name and path."
    else
        # Assuming conda init is already run and managed
        echo "Conda is initialized for RHEL9."
    fi
else
    echo "Miniconda3 directory not found on RHEL9."
fi

eval "$('/home/ghoyer/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# Your Conda environment name: 'myenv'
# CONDA_ENV_NAME=/data/VirtualAging/users/ghoyer/conda/envs/air

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
python src/obj_detection/val_msk.py "${CONFIG_NAME}"

## End-of-job summary, if running as a job
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID" 
                                          


# # run as sbatch val_yolo.sh