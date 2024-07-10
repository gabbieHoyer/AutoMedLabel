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

# -------------------- USER DEFINED PARAMS -------------------- #

# Define a list containing the base name of your config file without the path or extension
CONFIG_LIST=("OAI_T1_Thigh_plus")


# Your Conda environment name: 'myenv'
if [ -d /data/VirtualAging ] ;
then
  dVA="VirtualAging"
else
  dVA="virtualaging"
fi

CONDA_ENV_NAME=/data/$dVA/users/ghoyer/conda/envs/autolabel

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

    # Run your scripts
    echo "Starting data_standardization.py with ${CONFIG_NAME}"
    python3 preprocessing/data_standardization.py "${CONFIG_NAME}"

    echo "Starting nifti_viz.py with ${CONFIG_NAME}"
    python3 evaluation/visualization/nifti_viz.py "${CONFIG_NAME}"

    echo "Starting metadata_creation.py with ${CONFIG_NAME} for operation A"
    python3 preprocessing/metadata_creation.py "${CONFIG_NAME}" --operation A

    echo "Starting metadata_creation.py with ${CONFIG_NAME} for operation B"
    python3 preprocessing/metadata_creation.py "${CONFIG_NAME}" --operation B

    echo "Starting slice_standardization.py with ${CONFIG_NAME}"
    python3 preprocessing/slice_standardization.py "${CONFIG_NAME}"

    echo "Starting npy_viz.py with ${CONFIG_NAME}"
    python3 evaluation/visualization/npy_viz.py "${CONFIG_NAME}"

    echo "Starting metadata_creation.py with ${CONFIG_NAME} for operation C"
    python3 preprocessing/metadata_creation.py "${CONFIG_NAME}" --operation C

    # ----------------------- Bonus ---------------------- #
    # echo "Starting metadata_creation.py with ${CONFIG_NAME} for operation D"
    # python3 preprocessing/metadata_creation.py "${CONFIG_NAME}" --operation D

    echo "All processes completed successfully."

done 

# to run shell script: sbatch build_sam_dataset.sh 
# from inside the /scripts directory




# rhel_version=$(cat /etc/redhat-release)
# if [[ "$rhel_version" =~ "release 9" ]]; then
#     # RHEL9-specific initialization
#     export MODULEPATH=$MODULEPATH:/home/ghoyer/Modules/modulefiles
#     # Load custom user installations module path
#     module load use.own
#     if [ -d "/home/ghoyer/miniconda3" ]; then
#         # Load Conda module for RHEL9
#         module load conda_base/1.0
#         if [ $? -ne 0 ]; then
#             echo "Failed to load Miniconda module for RHEL9. Check module name and path."
#         else
#             # Assuming conda init is already run and managed
#             echo "Conda is initialized for RHEL9."
#         fi
#     else
#         echo "Miniconda3 directory not found on RHEL9."
#     fi

# # General Conda initialization (this will be run regardless of RHEL version)
# __conda_setup="$('/home/ghoyer/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/home/ghoyer/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/home/ghoyer/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/home/ghoyer/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup