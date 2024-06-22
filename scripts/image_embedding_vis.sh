#!/bin/bash

#SBATCH --job-name=vis_img_embed
#SBATCH --output=../logs/vis_img_embed_%j.log  # Save log to file with job ID
#SBATCH --mem=16G                       # Memory requirement
#SBATCH --time=05:00:00                 # Time limit hrs:min:sec
##SBATCH --nodelist=hyperion
#SBATCH --partition=gpu                # Specify the GPU partition
#SBATCH --gres=gpu:1

echo "Running on node $HOSTNAME"

# Activate conda environment
export PATH=/netopt/rhel7/bin:$PATH
eval "$('/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
CONDA_ENV_NAME=/data/VirtualAging/users/ghoyer/conda/envs/medsam

# List of configurations
# CONFIG_NAMES=("P50-AF" "TBrecon" "OAI_imorphics" "Bacpac_t1sag_mdai" "DHAL" "Bacpac_t1ax_mdai")
CONFIG_NAMES=("TBrecon")

# Activate Conda environment
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}


# Iterate through each configuration
for CONFIG_NAME in "${CONFIG_NAMES[@]}"; do
    # scripts dir the first time
    echo "The current directory is:"
    pwd
    
    echo "Processing configuration: $CONFIG_NAME"
    CONFIG_FILE="../config/zero_shot/${CONFIG_NAME}.yaml"

    echo "current config path: $CONFIG_FILE"

    # Check if the config file exists before proceeding
    if [ -f "$CONFIG_FILE" ]; then
        echo "Configuration file found: $CONFIG_FILE"

        cd ../notebooks || { echo "Failed to change directory to ../notebooks"; exit 1; }

        # Call Python script with the current configuration file
        python image_embeddings.py "$CONFIG_NAME"
        
        echo "Completed processing for configuration: $CONFIG_NAME"
    else
        echo "Error: Configuration file not found at $CONFIG_FILE"
    fi

    # scripts dir the first time
    echo "The current directory is:"
    pwd

    cd ../scripts || { echo "Failed to change directory to ../scripts"; exit 1; }

    # scripts dir the first time
    echo "The current directory is:"
    pwd
done

echo "All configurations processed successfully."

