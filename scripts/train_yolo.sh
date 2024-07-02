#!/bin/bash

# #SBATCH --job-name=yo
# #SBATCH --nodes=1
# #SBATCH --cpus-per-task=4
# #SBATCH --ntasks-per-node=2
# #SBATCH --time=0-22:00:00
# ##SBATCH --mem=16G
# #SBATCH --mem-per-cpu=64G
# #SBATCH --partition=dgx
# #SBATCH --gres=gpu:teslav100:2
# #SBATCH --output=../logs/%x-%j-yolov8_msk.out
# #SBATCH --error=../logs/yolov8_msk.err
# file


#!/bin/bash
#SBATCH --job-name=lols
#SBATCH --output=../logs/%x-%j-yolov8_msk.out
#SBATCH --error=../logs/yolov8_msk.err
#SBATCH --nodes=1                                    # Request 1 node
#SBATCH --time=20:00:00                              # Set a time limit for the job

#SBATCH --nodelist=juno
#SBATCH --partition=dgx
#SBATCH --gres=gpu:teslav100:4
#SBATCH --ntasks-per-node=4                          # Request 2 tasks (processes) per node
#SBATCH --mem=128G                                    # 32 GB VRAM per GPU; 16 GPUs; 96 CPU Cores; 24 Hrs max wall time
#SBATCH --cpus-per-task=6     


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


# Path to your config file
# Thigh 
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
python src/obj_detection/train_msk.py "${CONFIG_NAME}"

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
echo "All processes completed successfully."

# python train_air.py "${CONFIG_NAME}"

# # run as sbatch train_yolo.sh if in scripts dir, or
# # sbatch /path/to/scripts/train_yolo.sh
