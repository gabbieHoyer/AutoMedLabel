#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=../logs/finetune_multi_gpu_%j.log  # Save output to log file, %j will be replaced with the job ID
#SBATCH --nodes=1                                    # Request 1 node
#SBATCH --time=22:00:00                              # Set a time limit for the job

# #SBATCH --nodelist=anahita
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:4
# #SBATCH --ntasks-per-node=4                        
# #SBATCH --mem=132G                                    # 40 GB VRAM per GPU; 4 GPUs; 32 CPU Cores; 24 Hrs max wall time
# #SBATCH --cpus-per-task=8                            # Request 2 CPUs per task (useful for multi-threading)
# d

# #SBATCH --nodelist=juno
# #SBATCH --partition=dgx
# #SBATCH --gres=gpu:teslav100:4
# #SBATCH --ntasks-per-node=4                          # Request 2 tasks (processes) per node
# #SBATCH --mem=128G                                    # 32 GB VRAM per GPU; 16 GPUs; 96 CPU Cores; 24 Hrs max wall time
# #SBATCH --cpus-per-task=6                            # Request 2 CPUs per task (useful for multi-threading)
# g

# #SBATCH --nodelist=rhea                          
# #SBATCH --partition=gpu  
# #SBATCH --ntasks-per-node=4                         # Request 2 tasks (processes) per node                            
# #SBATCH --gres=gpu:4                               # 32 GB VRAM per GPU; 4 GPUs; 64 CPU Cores; 48 Hrs max wall time
# #SBATCH --mem=128G 
# #SBATCH --cpus-per-task=8                            # Request 2 CPUs per task (useful for multi-threading)
# h

# #SBATCH --nodelist=hyperion
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:4
# #SBATCH --ntasks-per-node=4                          # Request 2 tasks (processes) per node
# #SBATCH --mem=96G                                    # 32 GB VRAM per GPU; 16 GPUs; 96 CPU Cores; 24 Hrs max wall time
# #SBATCH --cpus-per-task=6                            # Request 2 CPUs per task (useful for multi-threading)
# f

#SBATCH --nodelist=hyperion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1                          # Request 2 tasks (processes) per node
#SBATCH --mem=24G                                    # 32 GB VRAM per GPU; 16 GPUs; 96 CPU Cores; 24 Hrs max wall time
#SBATCH --cpus-per-task=6                            # Request 2 CPUs per task (useful for multi-threading)


echo "Running on node $HOSTNAME"

echo "The current directory is:"
pwd

echo "chmod set_env.sh:"
# Ensure the environment setup script is executable
chmod +x env_scripts/set_env.sh

echo "The current directory is:"
pwd

echo "activate set_env.sh:"
# Source the script to set the environment
source env_scripts/set_env.sh

# After sourcing your environment script
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "DIST_URL=$DIST_URL"

# Activate conda environment
export PATH=/netopt/rhel7/bin:$PATH
eval "$('/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"

# Your Conda environment name: 'myenv'
CONDA_ENV_NAME=/data/VirtualAging/users/ghoyer/conda/envs/medsam

# Path to your config file  
CONFIG_NAME="OAI_Knee"

echo "config $CONFIG_NAME chosen for finetuning pipeline"

## Activate Conda environment
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}

## Navigate to root directory
cd .. || exit


echo "Running distributed training 5 with torchrun"
torchrun --nproc_per_node=$GPUS_PER_NODE \
         --nnodes=$NNODES \
         --node_rank=$RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         src/finetuning/finetune_evaluate.py ${CONFIG_NAME} 

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"


echo "All processes completed successfully."