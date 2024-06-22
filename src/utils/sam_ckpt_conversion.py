# -*- coding: utf-8 -*-
import torch
from collections import OrderedDict
import os

# sam_ckpt_path = "/data/TBrecon3/Users/ghoyer/SAM_data/embedding_exploration/mskSAM-miniTBreconSets-20231106-1957/0_ep/sam_vit_b_01ec64.pth"

# Add lists of paths here
finetuned_ckpt_paths = [
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_5_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_4/20240421-1552_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/P50_AF/SAM_20_trainSubjects_sliceBalance_True_imgEnc_${module.freeze.image_encoder}_maskDec_${module.freeze.mask_decoder}/Run_8/20240424-0943_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_True_imgEnc_True_maskDec_True/Run_4/20240423-2211_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/DHAL-Shoulder/SAM_5_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_2/20240421-0045_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/OAI-Thigh/SAM_40_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_5/20240423-2120_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/BACPAC_T1_sagittal/SAM_5_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_2/20240425-1312_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/BACPAC_t2_axial/SAM_20_trainSubjects_sliceBalance_False_imgEnc_${module.freeze.image_encoder}_maskDec_${module.freeze.mask_decoder}/Run_2/20240425-0027_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/BACPAC_t1_axial/SAM_5_trainSubjects_sliceBalance_False_imgEnc_${module.freeze.image_encoder}_maskDec_${module.freeze.mask_decoder}/Run_6/20240424-2051_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/Spine/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_4/20240427-0452_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/mskMuscle/SAM_5_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_5/20240427-0459_finetuned_model_best.pth",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/OAI-Thigh/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_3/20240515-2234_finetuned_model_best.pth"
#"/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/OAI-Thigh/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_4/20240516-0055_finetuned_model_best.pth"
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/Spine_T1ax/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_2/20240522-0442_finetuned_model_best.pth"

# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/OAI-Knee/SAM_40_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240526-0748_finetuned_model_latest_epoch_15.pth" # corrupted
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_20_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_3/20240610-1336_finetuned_model_latest_epoch_20.pth"
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/OAI-Knee/SAM_40_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240526-0748_finetuned_model_latest_epoch_10.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_20_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_4/20240611-1314_finetuned_model_latest_epoch_30.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_2/20240610-2015_finetuned_model_latest_epoch_25.pth"
#"/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_full_trainSubjects_sliceBalance_True_imgEnc_True_maskDec_True/Run_3/20240613-1326_finetuned_model_latest_epoch_10.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_3/20240613-2016_finetuned_model_latest_epoch_45.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_1/20240607-2345_finetuned_model_latest_epoch_5.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_4/20240614-0021_finetuned_model_latest_epoch_15.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240611-1830_finetuned_model_latest_epoch_15.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_full_trainSubjects_sliceBalance_True_imgEnc_True_maskDec_True/Run_4/20240614-1354_finetuned_model_best.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_full_trainSubjects_sliceBalance_True_imgEnc_False_maskDec_True/Run_3/20240611-2041_finetuned_model_latest_epoch_20.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_3/20240612-0032_finetuned_model_latest_epoch_0.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_2/20240529-1430_finetuned_model_latest_epoch_5.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_2/20240608-2328_finetuned_model_latest_epoch_0.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240608-2328_finetuned_model_latest_epoch_0.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_3/20240614-2208_finetuned_model_latest_epoch_20.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_6/20240614-2306_finetuned_model_latest_epoch_25.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_4/20240615-2055_finetuned_model_latest_epoch_35.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_1/20240616-1954_finetuned_model_latest_epoch_5.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_2/20240617-1449_finetuned_model_latest_epoch_0.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_3/20240616-1514_finetuned_model_latest_epoch_15.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_3/20240617-2320_finetuned_model_latest_epoch_15.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_4/20240618-0855_finetuned_model_latest_epoch_15.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_6/20240619-1118_finetuned_model_latest_epoch_20.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_5/20240618-2305_finetuned_model_latest_epoch_25.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_full_trainSubjects_sliceBalance_True_imgEnc_False_maskDec_True/Run_2/20240527-2256_finetuned_model_latest_epoch_5.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_4/20240620-1509_finetuned_model_latest_epoch_5.pth"

"/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_8/20240620-1504_finetuned_model_latest_epoch_35.pth"

]

save_paths = [
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/TBrecon_Knee",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/P50_AFACL_Knee",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/TBrecon_OAI_P50_AFACL_Knee",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/DHAL_Shoulder",
#"/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/OAI_Thigh/rand_single_instance_gt_matched",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/UH2_T1sag",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/UH2_T2ax",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/UH2_T1ax",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/Spine_UH2",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/Challenging_Anatomy",
# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/UH2_T1ax"

# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/model_weights/Finetuned/UH2_T1sag"

# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/OAI-Knee/SAM_40_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240526-0748_finetuned_model_latest_epoch_15_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_20_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_3/20240610-1336_finetuned_model_latest_epoch_20_converted.pth"

# "/data/VirtualAging/users/ghoyer/correcting_rad_workflow/segmentation/mskSam/work_dir/finetuning/OAI-Knee/SAM_40_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240526-0748_finetuned_model_latest_epoch_10_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_20_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_4/20240611-1314_finetuned_model_latest_epoch_30_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_2/20240610-2015_finetuned_model_latest_epoch_25_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_full_trainSubjects_sliceBalance_True_imgEnc_True_maskDec_True/Run_3/20240613-1326_finetuned_model_latest_epoch_10_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_3/20240613-2016_finetuned_model_latest_epoch_45_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_1/20240607-2345_finetuned_model_latest_epoch_5_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_4/20240614-0021_finetuned_model_latest_epoch_15_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240611-1830_finetuned_model_latest_epoch_15_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_full_trainSubjects_sliceBalance_True_imgEnc_True_maskDec_True/Run_4/20240614-1354_finetuned_model_best_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_full_trainSubjects_sliceBalance_True_imgEnc_False_maskDec_True/Run_3/20240611-2041_finetuned_model_latest_epoch_20_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_3/20240612-0032_finetuned_model_latest_epoch_0_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_2/20240529-1430_finetuned_model_latest_epoch_5_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_2/20240608-2328_finetuned_model_latest_epoch_0_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_1/20240608-2328_finetuned_model_latest_epoch_0_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_3/20240614-2208_finetuned_model_latest_epoch_20_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_40_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_6/20240614-2306_finetuned_model_latest_epoch_25_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_4/20240615-2055_finetuned_model_latest_epoch_35_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_1/20240616-1954_finetuned_model_latest_epoch_5_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_2/20240617-1449_finetuned_model_latest_epoch_0_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_3/20240616-1514_finetuned_model_latest_epoch_15_converted.pth"


# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_3/20240617-2320_finetuned_model_latest_epoch_15_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_4/20240618-0855_finetuned_model_latest_epoch_15_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_6/20240619-1118_finetuned_model_latest_epoch_20_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_5/20240618-2305_finetuned_model_latest_epoch_25_converted.pth"

# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_full_trainSubjects_sliceBalance_True_imgEnc_False_maskDec_True/Run_2/20240527-2256_finetuned_model_latest_epoch_5_converted.pth"
# "/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/OAI-P50-TBrecon-AFACL/SAM_full_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_4/20240620-1509_finetuned_model_latest_epoch_5_converted.pth"

"/data/mskprojects/mskSAM/users/ghoyer/mskSam/work_dir/finetuning/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_8/20240620-1504_finetuned_model_latest_epoch_35_converted.pth"

]

# Function to check if the 'module.' prefix is present
def check_module_prefix(state_dict):
    return any(k.startswith('module.') for k in state_dict.keys())

# Function to process and save a single checkpoint
def process_and_save_checkpoint(finetuned_ckpt_path, save_dir):
    # Extract the filename from the finetuned_ckpt_path
    filename = os.path.basename(finetuned_ckpt_path)
    
    # Construct the full save path with the original filename
    # full_save_path = os.path.join(save_dir, filename) if os.path.isdir(save_dir) else save_dir
    
    # temp change
    full_save_path = save_dir

    # Load the fine-tuned checkpoint
    finetuned_ckpt = torch.load(finetuned_ckpt_path)
    
    # Correct the 'model' keys if the checkpoint was saved from a multi-GPU setup
    if 'model' in finetuned_ckpt and check_module_prefix(finetuned_ckpt['model']):
        new_model_state_dict = OrderedDict()
        for k, v in finetuned_ckpt['model'].items():
            new_key = k[7:] if k.startswith('module.') else k  # remove `module.`
            new_model_state_dict[new_key] = v
        finetuned_ckpt['model'] = new_model_state_dict
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    
    # Save the updated checkpoint
    torch.save(finetuned_ckpt, full_save_path)

# Iterate over the pairs of paths and process them
for finetuned_ckpt_path, save_path in zip(finetuned_ckpt_paths, save_paths):
    process_and_save_checkpoint(finetuned_ckpt_path, save_path)
