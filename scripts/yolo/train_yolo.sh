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
#SBATCH --time=22:00:00                              # Set a time limit for the job

#SBATCH --nodelist=juno
#SBATCH --partition=dgx
#SBATCH --gres=gpu:teslav100:4
#SBATCH --ntasks-per-node=4                          # Request 2 tasks (processes) per node
#SBATCH --mem=128G                                    # 32 GB VRAM per GPU; 16 GPUs; 96 CPU Cores; 24 Hrs max wall time
#SBATCH --cpus-per-task=6     


# #SBATCH --nodelist=rhea                          
# #SBATCH --partition=gpu  
# #SBATCH --ntasks-per-node=4                         # Request 2 tasks (processes) per node                            
# #SBATCH --gres=gpu:4                               # 32 GB VRAM per GPU; 4 GPUs; 64 CPU Cores; 48 Hrs max wall time
# #SBATCH --mem=128G 
# #SBATCH --cpus-per-task=8                            # Request 2 CPUs per task (useful for multi-threading)
# f

# #SBATCH --nodelist=rhea                          
# #SBATCH --partition=gpu  
# #SBATCH --ntasks-per-node=3                         # Request 2 tasks (processes) per node                            
# #SBATCH --gres=gpu:3                               # 32 GB VRAM per GPU; 4 GPUs; 64 CPU Cores; 48 Hrs max wall time
# #SBATCH --mem=96G 
# #SBATCH --cpus-per-task=8                            # Request 2 CPUs per task (useful for multi-threading)
# d

# #SBATCH --nodelist=hyperion
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:3
# #SBATCH --ntasks-per-node=3                          # Request 2 tasks (processes) per node
# #SBATCH --mem=72G                                    # 32 GB VRAM per GPU; 16 GPUs; 96 CPU Cores; 24 Hrs max wall time
# #SBATCH --cpus-per-task=6                            # Request 2 CPUs per task (useful for multi-threading)
# g

# #SBATCH --nodelist=anahita
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:4
# #SBATCH --ntasks-per-node=4                        
# #SBATCH --mem=132G                                    # 40 GB VRAM per GPU; 4 GPUs; 32 CPU Cores; 24 Hrs max wall time
# #SBATCH --cpus-per-task=8   
# g

# Activate conda environment

if [ -d /data/VirtualAging ] ;
then
  dVA="VirtualAging"
else
  dVA="virtualaging"
fi

CONDA_ENV_NAME=/data/$dVA/users/ghoyer/conda/envs/autolabel


# Path to your config file
# Thigh 
# CONFIG_NAME="OAI_Thigh_yolo"
# Knee
# CONFIG_NAME="OAI_imorphics_yolo"
# Spine
# CONFIG_NAME="UH3_spine_t1sag_yolo"

CONFIG_NAME="DHAL_shoulder_yolo"

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
