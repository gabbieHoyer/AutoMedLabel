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
export PATH=/netopt/rhel7/bin:$PATH
eval "$('/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"

# Your Conda environment name: 'myenv'
CONDA_ENV_NAME=/data/VirtualAging/users/ghoyer/conda/envs/air

# Path to your config file

# CONFIG_NAME="exp1_obj_det"
# CONFIG_NAME="exp2_obj_det"

# CONFIG_NAME="exp1_obb"
# CONFIG_NAME="exp1_seg"  # Update this with your actual config name without the extension
# CONFIG_NAME="exp1_class"

# Thigh 
CONFIG_NAME="OAI_Thigh2"

# Activate Conda environment
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}

# Print node in the terminal
echo "Running on node $HOSTNAME"

# Navigate to the script's directory, then to the root directory
# cd /data/VirtualAging/users/ghoyer/correcting_rad_workflow/detection/VLS/AirDet
# cd /data/virtualaging/users/ghoyer/correcting_rad_workflow/det2seg/AutoMedLabel

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
