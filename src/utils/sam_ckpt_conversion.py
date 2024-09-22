# -*- coding: utf-8 -*-
import torch
from collections import OrderedDict
import os

# Add lists of paths here
finetuned_ckpt_paths = [
# "/mskSam/work_dir/finetuning/TBrecon-Knee/SAM_5_trainSubjects_sliceBalance_False_imgEnc_True_maskDec_True/Run_4/20240421-1552_finetuned_model_best.pth",
"/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_8/20240620-1504_finetuned_model_latest_epoch_35.pth"

]

save_paths = [
# "/mskSam/work_dir/model_weights/Finetuned/TBrecon_Knee",
"/mskSAM/SAM_full_trainSubjects_sliceBalance_False_imgEnc_False_maskDec_True/Run_8/20240620-1504_finetuned_model_latest_epoch_35_converted.pth"

]

# Function to check if the 'module.' prefix is present
def check_module_prefix(state_dict):
    return any(k.startswith('module.') for k in state_dict.keys())

# Function to process and save a single checkpoint
def process_and_save_checkpoint(finetuned_ckpt_path, save_dir):
    # Extract the filename from the finetuned_ckpt_path
    filename = os.path.basename(finetuned_ckpt_path)
    
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
