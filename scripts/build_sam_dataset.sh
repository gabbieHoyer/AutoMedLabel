#!/bin/bash

#SBATCH --job-name=build_dataset
#SBATCH --output=../logs/build_dataset_%j.log  # Save log to file with job ID
#SBATCH --mem=16G                       # Memory requirement
#SBATCH --time=02:00:00                 # Time limit hrs:min:sec
#SBATCH --cpus-per-task=4               # Number of CPU cores per task

# Enable exit on error
set -e

echo "Running on node $HOSTNAME"

# Activate conda environment
export PATH=/netopt/rhel7/bin:$PATH
eval "$('/netopt/rhel7/versions/python/Anaconda3-edge/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"

# --------------------
# CONFIG OPTIONS
# --------------------
##### INCOMPLETE PROCESSING OPTIONS #####
# Knee
CONFIG_LIST=("OAI_zib") # Completed NIfTI, but going to move forward with OAI_imporphics
CONFIG_LIST=("FreeMax") 
# Spine
# Other anatomy

##### IN PROG OPTIONS #####
# Knee
CONFIG_LIST=("P50_binary" "P50_compart" "AFACL_binary" "AFACL_compart") 
# Spine
CONFIG_LIST=( "BACPAC_MDai_T1ax") 
# Other anatomy
CONFIG_LIST=("KICK_cube")

##### COMPLETED OPTIONS #####
# Knee
CONFIG_LIST=("TBrecon" "K2S" "OAI_imorphics") 
# Spine
CONFIG_LIST=("BACPAC_MDai_T1sag" "BACPAC_MDai_T1sag_vert" "BACPAC_MDai_T2ax" "BACPAC_T1sag" "BACPAC_DL_T1sag" "BACPAC_DL_T2ax") 
# Other anatomy
CONFIG_LIST=("OAI_T1_thigh" "DHAL")

# -------------------- USER DEFINED PARAMS -------------------- #

# Define a list containing the base name of your config file without the path or extension
#"BACPAC_MDai_T1sag" "BACPAC_MDai_T1sag_vert" "BACPAC_MDai_T2ax" "BACPAC_T1sag"
CONFIG_LIST=( "BACPAC_T2ax" "BACPAC_DL_T2ax")  #"BACPAC_DL_T2ax") 

CONFIG_LIST=("UH3_T2ax")

# Your Conda environment name: 'myenv'
CONDA_ENV_NAME=/data/VirtualAging/users/ghoyer/conda/envs/medsam


# -------------------- BEGIN CODE -------------------- #

## Activate Conda environment
echo "Activating Conda environment: ${CONDA_ENV_NAME}"
source activate ${CONDA_ENV_NAME} || conda activate ${CONDA_ENV_NAME}

## Navigate to the data processing directory
cd ../src || exit

## Iterate through each dataset
for CONFIG_NAME in "${CONFIG_LIST[@]}"; do 
    echo "**************************************************"
    echo "config $CONFIG_NAME chosen for data-processing pipeline"

    # ## Run your scripts
    # echo "Starting data_standardization.py with ${CONFIG_NAME}"
    # python3 preprocessing/data_standardization.py "${CONFIG_NAME}"

    # echo "Starting nifti_viz.py with ${CONFIG_NAME}"
    # python3 evaluation/visualizing/nifti_viz.py "${CONFIG_NAME}"

    # echo "Starting metadata_creation.py with ${CONFIG_NAME} for operation A"
    # python3 preprocessing/metadata_creation.py "${CONFIG_NAME}" --operation A

    # echo "Starting metadata_creation.py with ${CONFIG_NAME} for operation B"
    # python3 preprocessing/metadata_creation.py "${CONFIG_NAME}" --operation B

    # echo "Starting slice_standardization.py with ${CONFIG_NAME}"
    # python3 preprocessing/slice_standardization.py "${CONFIG_NAME}"

    # echo "Starting npy_viz.py with ${CONFIG_NAME}"
    # python3 evaluation/visualizing/npy_viz.py "${CONFIG_NAME}"

    echo "Starting metadata_creation.py with ${CONFIG_NAME} for operation C"
    python3 preprocessing/metadata_creation.py "${CONFIG_NAME}" --operation C

    echo "All processes completed successfully."

done 
# to run shell script: sbatch build_sam_dataset.sh 
# from inside the /scripts directory
