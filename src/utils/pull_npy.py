import os
import shutil
import json

# Define the directories
old_img_dir = '/data/mskscratch/standardized/Test_2025/OAI/thigh/T1_ax/npy/imgs'

old_labels_dir = '/data/mskscratch/standardized/Test_2025/OAI/thigh/T1_ax/npy/labels'

metadata_path = '/data/mskscratch/standardized/Test_2025/OAI/thigh/T1_ax/metadata/volume_metadata_for_ml.json'

new_base_dir = '/data/mskscratch/standardized/Test_2025/OAI/thigh/T1_ax/npy/YOLO'

# Load metadata
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Create new directories if they don't exist
splits = ['train', 'val', 'test']
subfolders = ['images', 'labels']
for split in splits:
    for subfolder in subfolders:
        os.makedirs(os.path.join(new_base_dir, split, subfolder), exist_ok=True)

# Function to copy files
def copy_files(subject_id, split, old_dir, new_dir):
    for file_name in os.listdir(old_dir):
        if file_name.startswith(subject_id):
            src = os.path.join(old_dir, file_name)
            dst = os.path.join(new_base_dir, split, new_dir, file_name)
            shutil.copy2(src, dst)

# Copy files based on metadata
for subject_id, data in metadata.items():
    split = data.get('Split')
    if split in splits:
        # copy_files(subject_id, split, old_img_dir, 'images')
        copy_files(subject_id, split, old_labels_dir, 'labels')

print("Files copied successfully.")
