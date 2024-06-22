
import json
import torch
import numpy as np
from abc import ABC
from torch.nn import Module
from torchmetrics import Metric
from monai.metrics import Metric
from monai.utils import MetricReduction
from monai.metrics.utils import do_metric_reduction

from skimage import transform
from src.preprocessing.sam_prep import MaskPrep #, ImagePrep

# -------------------- DATA POST-PROCESS FUNCTIONS -------------------- #

# monai post-processing functions: 
# https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/post/array.py#L132
# fill in holes, keep largest connected components, asDiscrete

# __all__ = [
#     "Activations",
#     "AsDiscrete",
#     "FillHoles",
#     "KeepLargestConnectedComponent",
#     "RemoveSmallObjects",
#     "LabelFilter",
#     "LabelToContour",
#     "MeanEnsemble",
#     "ProbNMS",
#     "SobelGradients",
#     "VoteEnsemble",
#     "Invert",
#     "DistanceTransformEDT",
# ]

# ----------------------------------------------------------------------- #
  

# def postprocess_resize(mask, image_size_tuple:tuple[int,int]):
#     """Resize mask to new dimensions."""
#     predMaskPrep = MaskPrep()
#     resized_mask = predMaskPrep.resize_mask(mask_data = mask.astype(np.uint8),
#                                             image_size_tuple = image_size_tuple)
#     return resized_mask

# or example of using the post-processing resize for a slice:
# gt_2D_prep_orig_size = postprocess_resize(gt_2D, (gt_2D_unprocessed.shape[0], gt_2D_unprocessed.shape[1]))

# def postprocess_prediction(sam_pred, image_size_tuple:tuple[int,int], label_id:int):
#     """Convert SAM prediction into segmentation mask with original image dims and label ids."""
#     sam_mask_resized = postprocess_resize(sam_pred, image_size_tuple)
#     sam_mask = np.zeros_like(sam_mask_resized, dtype=np.uint8)
#     sam_mask[sam_mask_resized > 0] = label_id
#     return sam_mask

# ----------------------------------------------------------------------------------- #

class TissueVolumeMetric(Module):
    def __init__(self, reduction='mean_batch', class_names=None, tissue_labels=None):
        super().__init__()
        self.reduction = reduction
        self.class_names = class_names if class_names else ['Class{}'.format(i) for i in range(1)]
        self.num_classes = len(self.class_names)
        self.tissue_labels = tissue_labels if tissue_labels is not None else list(range(self.num_classes))
        self.volumes_pred = {}
        self.volumes_true = {}

    def reset(self):
        self.volumes_pred = {}
        self.volumes_true = {}

    def resize_mask(self, mask_slice, subject_slice_meta):

        # multiplying by 2 b/c I combined two mri into a single image loools
        image_size_tuple = [subject_slice_meta['rows'], subject_slice_meta['columns']]
        
        resized_mask = transform.resize(
            mask_slice,
            image_size_tuple,
            order=0,  # nearest-neighbor interpolation to preserve label integrity
            preserve_range=True,
            mode='constant',
            anti_aliasing=False
        )   
        return resized_mask

    def update(self, y_pred, y, subject_slice_meta, subj_id, slice_id):
        voxel_area = subject_slice_meta['pixel_spacing'][0] * subject_slice_meta['pixel_spacing'][1] * subject_slice_meta['slice_thickness'] / 1000.0

        # Resize the masks for each class (channel)
        y_pred_resized = np.stack([self.resize_mask(y_pred[0, i, :, :].cpu().numpy(), subject_slice_meta) for i in range(self.num_classes)], axis=0)
        y_resized = np.stack([self.resize_mask(y[0, i, :, :].cpu().numpy(), subject_slice_meta) for i in range(self.num_classes)], axis=0)

        # Add batch dimension back
        y_pred_resized = np.expand_dims(y_pred_resized, axis=0)
        y_resized = np.expand_dims(y_resized, axis=0)

        # print(f"y_pred shape: {y_pred.shape}") #torch.Size([1, 11, 1024, 1024])
        # print(f"y_pred_resized shape: {y_pred_resized.shape}")

        # Create a unique key for each subject-slice pair
        subject_slice_key = f"{subj_id}_{slice_id}"

        if subject_slice_key not in self.volumes_pred:
            self.volumes_pred[subject_slice_key] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class
            self.volumes_true[subject_slice_key] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class

        total_volume_pred = 0.0
        total_volume_true = 0.0

        for i in range(self.num_classes):
            class_mask_pred = y_pred_resized[:, i, :, :].sum().item()  # Assuming y_pred is BxCxHxW
            class_mask_true = y_resized[:, i, :, :].sum().item()

            volume_pred = class_mask_pred * voxel_area
            volume_true = class_mask_true * voxel_area

            self.volumes_pred[subject_slice_key][i] += volume_pred
            self.volumes_true[subject_slice_key][i] += volume_true

            if i in self.tissue_labels:
                total_volume_pred += volume_pred
                total_volume_true += volume_true

        # Update the 'total' class
        self.volumes_pred[subject_slice_key][-1] += total_volume_pred
        self.volumes_true[subject_slice_key][-1] += total_volume_true

    def compute(self):
        # Fill with NaN for labels not in tissue_labels
        for key in self.volumes_pred:
            for i in range(self.num_classes):
                if i not in self.tissue_labels:
                    self.volumes_pred[key][i] = np.nan
                    self.volumes_true[key][i] = np.nan

        return self.volumes_pred, self.volumes_true

    def aggregate_by_subject(self):
        aggregated_volumes_pred = {}
        aggregated_volumes_true = {}

        # Aggregate predicted volumes
        for key, volumes in self.volumes_pred.items():
            subj_id = key[:-4]
            if subj_id not in aggregated_volumes_pred:
                aggregated_volumes_pred[subj_id] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class
            for i in range(self.num_classes):
                if i in self.tissue_labels:
                    aggregated_volumes_pred[subj_id][i] += volumes[i]
                else:
                    aggregated_volumes_pred[subj_id][i] = np.nan
            # Aggregate the 'total' class
            aggregated_volumes_pred[subj_id][-1] += volumes[-1]

        # Aggregate true volumes
        for key, volumes in self.volumes_true.items():
            subj_id = key[:-4]
            if subj_id not in aggregated_volumes_true:
                aggregated_volumes_true[subj_id] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class
            for i in range(self.num_classes):
                if i in self.tissue_labels:
                    aggregated_volumes_true[subj_id][i] += volumes[i]
                else:
                    aggregated_volumes_true[subj_id][i] = np.nan
            # Aggregate the 'total' class
            aggregated_volumes_true[subj_id][-1] += volumes[-1]

        return aggregated_volumes_pred, aggregated_volumes_true

    def forward(self, *args, **kwargs):
        # Implement the forward method as a placeholder
        pass


# ----------------------------------


# works for thigh - just got luck that npy slice vals do match subject id vals, hence can
# directly link dicom slice data back to sam data at this point

# other datasets may have different dicom slice naming convention, may have had empty slice values
# removed, and there does not exist a direct link from npy slices to this stats metadata b/c having
# a direct downstream task was not intended. However, if model inference was run fresh on the nifti test
# set volumes then there is flexiblitiy to keep track of those slices which may be removed (ignored by sam), and 
# you can leave resultant slices back to their dicom slice source. 

# future design consideration for biomarker metric needs / individual user needs 
# NOTE: if subject volume slices vary across the volume and those values are necessary for a particular metric
# if not, and you can guarantee consistency, then I suppose my alternative choice (subject slice representations - 
# one slice pulled per subject to use for all subsequent slices) may indeed suffice.

# def load_meta(metadata_path):
#     with open(metadata_path, 'r') as file:
#         metadata = json.load(file)
#     # Reduce the loaded metadata to only what is necessary to minimize memory usage
#     reduced_metadata = {subj_id: subject_data['slices'] for subj_id, subject_data in metadata.items()}
#     return reduced_metadata

# goal - make it so every metadataset[subj_id]['slices'][slice_num/instance_num] instead of slice name :/
# 'instanceNumber' should match the slice num in the npy file names
# def load_meta(metadata_path):
#     with open(metadata_path, 'r') as file:
#         metadata = json.load(file)
#     # Reduce the loaded metadata to only what is necessary to minimize memory usage
#     reduced_metadata = {subj_id: subject_data['slices'] for subj_id, subject_data in metadata.items()}
#     return reduced_metadata

# def load_meta(metadata_path):
#     with open(metadata_path, 'r') as file:
#         metadata = json.load(file)
    
#     import pdb; pdb.set_trace()

#     # Reduce the loaded metadata to only what is necessary to minimize memory usage
#     reduced_metadata = {}
#     for subj_id, subject_data in metadata.items():
#         slices = subject_data['slices']
#         # Create a new dictionary for each subject with instanceNumber as the key
#         reduced_slices = {str(slice_data['instanceNumber']).zfill(3): slice_data for slice_data in slices}
#         reduced_metadata[subj_id] = reduced_slices

#     import pdb; pdb.set_trace()
#     return reduced_metadata

# def parse_image_name(img_name):
#     """Parses the image name to extract subject ID and slice ID, increments slice ID by 1."""
#     base_name = img_name.split('.')[0]  # Remove the file extension
#     subj_id, slice_num = base_name.split('-')  # Split by the '-' to separate the subject ID and slice number
#     # Increment slice number by 1 and zero-pad to three digits
#     incremented_slice_id = str(int(slice_num) + 1).zfill(3)
#     return subj_id, incremented_slice_id

# def parse_image_name(img_name):
#     """Parses the image name to extract subject ID and slice ID, increments slice ID by 1."""
#     base_name = img_name.split('.')[0]  # Remove the file extension

#     # Find the position of the last dash
#     last_dash_index = base_name.rfind('-')

#     # Split the base name at the last dash
#     subj_id = base_name[:last_dash_index]
#     slice_num = base_name[last_dash_index + 1:]  # Take the part after the last dash
   
#     # Increment slice number by 1 and zero-pad to three digits
#     incremented_slice_id = str(int(slice_num) + 1).zfill(3)
#     return subj_id, incremented_slice_id


# b/c slice id is for npy file name assignments but doesnt necessarily match the slice instance number from dicom (which has silly names depending on dataset)
# insrstance number shuld be in format 001 instead of 1 if its not that already
# def extract_meta(metadata_dict, subj_id, slice_id):
#     """Extracts metadata for a specific image slice"""

#     # import pdb; pdb.set_trace()

#     slice_meta = metadata_dict[subj_id][slice_id] # metadata starts at index 1; npy slices starts at index 0
    
#     pixel_spacing = slice_meta['pixel_spacing']
#     slice_thickness = slice_meta['slice_thickness']
#     return pixel_spacing, slice_thickness


# NOTE: can only guarantee this is useful for thigh - 
# may need a variation of volume computation depending on if 
# thigh differs across slices for necessary info and if other 
# datasets have non-significant variation across subject slices for 
# dicom fields of interest - may fully move to representative slice design

# def extract_meta(metadata_dict, subj_id, slice_id, fields):
#     """
#     Extracts metadata for a specific subject and returns the requested fields.
    
#     Args:
#         metadata_dict (dict): The metadata dictionary.
#         subj_id (str): The subject ID.
#         slice_id (str): The slice name ID.
#         fields (list): List of fields to extract.
        
#     Returns:
#         dict: A dictionary of the requested fields and their values.
#     """
#     subject_meta = metadata_dict.get(subj_id, {}).get(slice_id, {})
#     extracted_meta = {field: subject_meta.get(field) for field in fields}
#     return extracted_meta


# -------------------------------------------------------------------------------------------- #

# works! without resizing:

# class TissueVolumeMetric(Module, ABC):
#     def __init__(self, reduction='mean_batch', class_names=None, tissue_labels=None):
#         super().__init__()
#         self.reduction = reduction
#         self.class_names = class_names if class_names else ['Class{}'.format(i) for i in range(1)]
#         self.num_classes = len(self.class_names)
#         self.tissue_labels = tissue_labels if tissue_labels is not None else list(range(self.num_classes))
#         self.volumes_pred = {}
#         self.volumes_true = {}

#     def reset(self):
#         self.volumes_pred = {}
#         self.volumes_true = {}

#     def update(self, y_pred, y, subject_slice_meta, subj_id, slice_id):   #pixel_spacing, slice_thickness,
#         # voxel_area = pixel_spacing[0] * pixel_spacing[1] * slice_thickness / 1000.0
#         voxel_area = subject_slice_meta['pixel_spacing'][0] * subject_slice_meta['pixel_spacing'][1] * subject_slice_meta['slice_thickness'] / 1000.0

#         # Create a unique key for each subject-slice pair
#         subject_slice_key = f"{subj_id}_{slice_id}"

#         if subject_slice_key not in self.volumes_pred:
#             self.volumes_pred[subject_slice_key] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class
#             self.volumes_true[subject_slice_key] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class

#         total_volume_pred = 0.0
#         total_volume_true = 0.0

#         for i in range(self.num_classes):
#             class_mask_pred = y_pred[:, i, :, :].sum().item()  # Assuming y_pred is BxCxHxW
#             class_mask_true = y[:, i, :, :].sum().item()

#             volume_pred = class_mask_pred * voxel_area
#             volume_true = class_mask_true * voxel_area

#             self.volumes_pred[subject_slice_key][i] += volume_pred
#             self.volumes_true[subject_slice_key][i] += volume_true

#             if i in self.tissue_labels:
#                 total_volume_pred += volume_pred
#                 total_volume_true += volume_true

#         # Update the 'total' class
#         self.volumes_pred[subject_slice_key][-1] += total_volume_pred
#         self.volumes_true[subject_slice_key][-1] += total_volume_true

#     def compute(self):
#         # Fill with NaN for labels not in tissue_labels
#         for key in self.volumes_pred:
#             for i in range(self.num_classes):
#                 if i not in self.tissue_labels:
#                     self.volumes_pred[key][i] = np.nan
#                     self.volumes_true[key][i] = np.nan

#         return self.volumes_pred, self.volumes_true

#     def aggregate_by_volume(self):
#         aggregated_volumes_pred = {}
#         aggregated_volumes_true = {}

#         # Aggregate predicted volumes
#         for key, volumes in self.volumes_pred.items():
#             subj_id = key[:-4]
#             if subj_id not in aggregated_volumes_pred:
#                 aggregated_volumes_pred[subj_id] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class
#             for i in range(self.num_classes):
#                 if i in self.tissue_labels:
#                     aggregated_volumes_pred[subj_id][i] += volumes[i]
#                 else:
#                     aggregated_volumes_pred[subj_id][i] = np.nan
#             # Aggregate the 'total' class
#             aggregated_volumes_pred[subj_id][-1] += volumes[-1]

#         # Aggregate true volumes
#         for key, volumes in self.volumes_true.items():
#             subj_id = key[:-4]
#             if subj_id not in aggregated_volumes_true:
#                 aggregated_volumes_true[subj_id] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class
#             for i in range(self.num_classes):
#                 if i in self.tissue_labels:
#                     aggregated_volumes_true[subj_id][i] += volumes[i]
#                 else:
#                     aggregated_volumes_true[subj_id][i] = np.nan
#             # Aggregate the 'total' class
#             aggregated_volumes_true[subj_id][-1] += volumes[-1]

#         return aggregated_volumes_pred, aggregated_volumes_true

#     def forward(self, *args, **kwargs):
#         # Implement the forward method as a placeholder
#         pass



# def resize_mask(self, mask_data, image_size_tuple:tuple[int,int]= None):
#     """
#     Resize mask data using nearest-neighbor interpolation to preserve label integrity.
#     Parameters:
#     - mask_data: The mask data as a numpy array.
#     - image_size_tuple: (H, W) in pixels to resize mask. 
    
#     Returns:
#     - A numpy array with dimensions (H, W).
#     """
#     def resize_mask_2D(mask_slice, image_size_tuple:tuple[int,int]):
#         # Resize the mask slice
#         resized_mask = transform.resize(
#             mask_slice,
#             image_size_tuple,
#             order=0,  # nearest-neighbor interpolation to preserve label integrity
#             preserve_range=True,
#             mode='constant',
#             anti_aliasing=False
#         )
#         return resized_mask
    
#     if image_size_tuple is None:
#         image_size_tuple = self.image_size_tuple

#     dims = len(np.shape(mask_data))
#     if dims == 2:
#         resized_masks = resize_mask_2D(mask_data, image_size_tuple)
#     elif dims == 3:
#         resized_masks = []
#         for mask_slice in mask_data:
#             resized_mask = resize_mask_2D(mask_slice, image_size_tuple)
#             resized_masks.append(resized_mask)
#         resized_masks = np.array(resized_masks)
    
#     return resized_masks

# ------------------------------------------------ #
    
# Voxel Volume Calculation:

# Voxel Volume = Pixel Spacing_x * Pixel Spacing_y * Slice Thickness
# Slice Area Contribution per Class:

# Slice Area Contribution = (Number of pixels in class) * Pixel Spacing_x * Pixel Spacing_y

# slice area contribution for each class in a plain text format:

# Slice Area Contribution = (Number of pixels in class) * Pixel Spacing_x * Pixel Spacing_y




# def resize_mask(mask_slice, subject_slice_meta):
#     """
#     Resize mask data using nearest-neighbor interpolation to preserve label integrity.
#     Parameters:
#     - mask_slice: The mask data as a numpy array.
#     - image_size_tuple: (H, W) in pixels to resize mask. 
    
#     Returns:
#     - A numpy array with dimensions (H, W).
#     """    

#     # image_size_tuple:tuple[int,int]= None
#     image_size_tuple = [subject_slice_meta['rows'], subject_slice_meta['columns']]

#     # gt2D slice mask shape -> torch.Size([1, 1, 1024, 1024]) # if not squeezed beforehand, else same shape as pred

#     # pred slice mask shape -> torch.Size([1, 1024, 1024])

#     dims = len(np.shape(mask_slice))
#     # if dims == 2:

#     # dang it - TypeError: can't convert cuda:0 device type tensor to numpy.
#     # Use Tensor.cpu() to copy the tensor to host memory first.
#     #
#     resized_mask = transform.resize(
#         mask_slice[0],
#         image_size_tuple,
#         order=0,  # nearest-neighbor interpolation to preserve label integrity
#         preserve_range=True,
#         mode='constant',
#         anti_aliasing=False
#     )
#     # resized_mask = transform.resize(
#     #     mask_slice,
#     #     image_size_tuple,
#     #     order=0,  # nearest-neighbor interpolation to preserve label integrity
#     #     preserve_range=True,
#     #     mode='constant',
#     #     anti_aliasing=False
#     # )
    
#     return resized_mask
    

# def postprocess_resize(mask, image_size_tuple:tuple[int,int]):
#     """Resize mask to new dimensions."""
#     predMaskPrep = MaskPrep()
#     resized_mask = predMaskPrep.resize_mask(mask_data = mask.astype(np.uint8),
#                                             image_size_tuple = image_size_tuple)
#     return resized_mask

# ---------------------------------------- works ---------------------------------------------------- #

# class CustomMetric(Module, ABC):
#     def __init__(self, reduction='mean_batch', class_names=None, tissue_labels=None):
#         super().__init__()
#         self.reduction = reduction
#         self.class_names = class_names if class_names else ['Class{}'.format(i) for i in range(1)]
#         self.num_classes = len(self.class_names)
#         self.tissue_labels = tissue_labels if tissue_labels is not None else list(range(self.num_classes))
#         self.volumes_pred = {}
#         self.volumes_true = {}

#     def reset(self):
#         self.volumes_pred = {}
#         self.volumes_true = {}

#     def update(self, y_pred, y, pixel_spacing, slice_thickness, subj_id, slice_id):
#         voxel_area = pixel_spacing[0] * pixel_spacing[1] * slice_thickness / 1000.0

#         # Create a unique key for each subject-slice pair
#         subject_slice_key = f"{subj_id}_{slice_id}"

#         if subject_slice_key not in self.volumes_pred:
#             self.volumes_pred[subject_slice_key] = [0.0] * self.num_classes
#             self.volumes_true[subject_slice_key] = [0.0] * self.num_classes

#         for i in range(self.num_classes):
#             class_mask_pred = y_pred[:, i, :, :].sum().item()  # Assuming y_pred is BxCxHxW
#             class_mask_true = y[:, i, :, :].sum().item()

#             volume_pred = class_mask_pred * voxel_area
#             volume_true = class_mask_true * voxel_area

#             self.volumes_pred[subject_slice_key][i] += volume_pred
#             self.volumes_true[subject_slice_key][i] += volume_true

#     def compute(self):
#         # Fill with NaN for labels not in tissue_labels
#         for key in self.volumes_pred:
#             for i in range(self.num_classes):
#                 if i not in self.tissue_labels:
#                     self.volumes_pred[key][i] = np.nan
#                     self.volumes_true[key][i] = np.nan

#         return self.volumes_pred, self.volumes_true

#     def aggregate_volumes_by_subject(self):
#         aggregated_volumes_pred = {}
#         aggregated_volumes_true = {}

#         # Aggregate predicted volumes
#         for key, volumes in self.volumes_pred.items():
#             subj_id = key[:-4]
#             if subj_id not in aggregated_volumes_pred:
#                 aggregated_volumes_pred[subj_id] = [0.0] * self.num_classes
#             for i in range(self.num_classes):
#                 if i in self.tissue_labels:
#                     aggregated_volumes_pred[subj_id][i] += volumes[i]
#                 else:
#                     aggregated_volumes_pred[subj_id][i] = np.nan

#         # Aggregate true volumes
#         for key, volumes in self.volumes_true.items():
#             subj_id = key[:-4]
#             if subj_id not in aggregated_volumes_true:
#                 aggregated_volumes_true[subj_id] = [0.0] * self.num_classes
#             for i in range(self.num_classes):
#                 if i in self.tissue_labels:
#                     aggregated_volumes_true[subj_id][i] += volumes[i]
#                 else:
#                     aggregated_volumes_true[subj_id][i] = np.nan

#         return aggregated_volumes_pred, aggregated_volumes_true

#     def forward(self, *args, **kwargs):
#         # Implement the forward method as a placeholder
#         pass

# --------------------------------------------------------------- #


# class CustomMetric(Module, ABC):
#     def __init__(self, reduction='mean_batch', class_names=None):
#         super().__init__()
#         self.reduction = reduction
#         self.class_names = class_names if class_names else ['Class{}'.format(i) for i in range(1)]
#         self.num_classes = len(self.class_names)
#         self.volumes_pred = {}
#         self.volumes_true = {}

#     def reset(self):
#         self.volumes_pred = {}
#         self.volumes_true = {}

#     def update(self, y_pred, y, pixel_spacing, slice_thickness, subj_id, slice_id):
#         # import pdb; pdb.set_trace()

#         voxel_area = pixel_spacing[0] * pixel_spacing[1] * slice_thickness / 1000.0

#         # Create a unique key for each subject-slice pair
#         subject_slice_key = f"{subj_id}_{slice_id}"

#         if subj_id not in self.volumes_pred:
#             self.volumes_pred[subject_slice_key] = [0.0] * self.num_classes
#             self.volumes_true[subject_slice_key] = [0.0] * self.num_classes

#         for i in range(self.num_classes):
#             # y_pred and y shape -> torch.Size([1, 11, 1024, 1024])
#             class_mask_pred = y_pred[:, i, :, :].sum().item()  # Assuming y_pred is BxCxHxW
#             class_mask_true = y[:, i, :, :].sum().item()

#             volume_pred = class_mask_pred * voxel_area
#             volume_true = class_mask_true * voxel_area

#             self.volumes_pred[subject_slice_key][i] += volume_pred
#             self.volumes_true[subject_slice_key][i] += volume_true

#     def compute(self):
#         return self.volumes_pred, self.volumes_true
    
#     def aggregate_volumes_by_subject(self):
#         aggregated_volumes_pred = {}
#         aggregated_volumes_true = {}

#         # Aggregate predicted volumes
#         for key, volumes in self.volumes_pred.items():
#             # subj_id = key.split('_')[0]
#             subj_id = key[:-4]
#             if subj_id not in aggregated_volumes_pred:
#                 aggregated_volumes_pred[subj_id] = [0.0] * self.num_classes
#             aggregated_volumes_pred[subj_id] = [sum(x) for x in zip(aggregated_volumes_pred[subj_id], volumes)]

#         # Aggregate true volumes
#         for key, volumes in self.volumes_true.items():
#             # subj_id = key.split('_')[0]
#             subj_id = key[:-4]
#             if subj_id not in aggregated_volumes_true:
#                 aggregated_volumes_true[subj_id] = [0.0] * self.num_classes
#             aggregated_volumes_true[subj_id] = [sum(x) for x in zip(aggregated_volumes_true[subj_id], volumes)]

#         return aggregated_volumes_pred, aggregated_volumes_true

#     def forward(self, *args, **kwargs):
#         # Implement the forward method as a placeholder
#         pass


# class CustomMetric(Metric):
#     def __init__(self, reduction='mean_batch', num_classes=None, class_names=None):
#         super().__init__()
#         self.reduction = reduction
#         self.num_classes = num_classes
#         self.class_names = class_names if class_names else ['Class{}'.format(i) for i in range(num_classes)]
#         self.volumes_pred = {}
#         self.volumes_true = {}

#     def reset(self):
#         self.volumes_pred = {}
#         self.volumes_true = {}

#     def update(self, y_pred, y, pixel_spacing, slice_thickness, subj_id):
#         y_pred, y = y_pred.to(y.device), y.to(y.device)
#         voxel_area = pixel_spacing[0] * pixel_spacing[1] * slice_thickness / 1000.0

#         if subj_id not in self.volumes_pred:
#             self.volumes_pred[subj_id] = [0.0] * self.num_classes
#             self.volumes_true[subj_id] = [0.0] * self.num_classes

#         for i in range(1, self.num_classes):
#             class_mask_pred = (y_pred == i).sum().item()
#             class_mask_true = (y == i).sum().item()

#             volume_pred = class_mask_pred * voxel_area
#             volume_true = class_mask_true * voxel_area

#             self.volumes_pred[subj_id][i-1] += volume_pred
#             self.volumes_true[subj_id][i-1] += volume_true

#     def compute(self):
#         return self.volumes_pred, self.volumes_true



# class CustomMetric(Metric):
#     def __init__(self, reduction='mean_batch', num_classes=None):
#         super().__init__()
#         self.reduction = reduction
#         self.num_classes = num_classes
#         self.volumes_pred = {}  # Store volumes on CPU to save GPU memory
#         self.volumes_true = {}  # Store volumes on CPU to save GPU memory

#     def reset(self):
#         self.volumes_pred = {}
#         self.volumes_true = {}

#     def update(self, y_pred, y, pixel_spacing, slice_thickness, subj_id):
#         y_pred, y = y_pred.to(y.device), y.to(y.device)

#         # Calculate the slice volume off GPU
#         voxel_area = pixel_spacing[0] * pixel_spacing[1]
#         slice_volume = voxel_area * slice_thickness

#         # Ensure we are not storing large data on GPU
#         if subj_id not in self.volumes_pred:
#             self.volumes_pred[subj_id] = torch.zeros(self.num_classes - 1, device='cpu')  # Ignore background class
#             self.volumes_true[subj_id] = torch.zeros(self.num_classes - 1, device='cpu')

#         for i in range(1, self.num_classes):
#             class_mask_pred = (y_pred == i).sum().item()  # Convert to Python scalar immediately
#             class_mask_true = (y == i).sum().item()

#             volume_pred = class_mask_pred * slice_volume
#             volume_true = class_mask_true * slice_volume

#             self.volumes_pred[subj_id][i-1] += volume_pred
#             self.volumes_true[subj_id][i-1] += volume_true

#         # Free up GPU memory
#         del y_pred, y
#         torch.cuda.empty_cache()  # Optionally clear any cached memory to prevent fragmentation

#     def compute(self):
#         # Return stored volumes
#         return self.volumes_pred, self.volumes_true




# class CustomMetric(Metric):
#     def __init__(self, reduction=MetricReduction.MEAN, num_classes=None):
#         super().__init__()
#         self.reduction = reduction
#         self.get_not_nans = False
#         self.num_classes = num_classes
#         self.scores = []

#     def reset(self):
#         self.scores = []

#     def update(self, y_pred, y):
#         # Ensure input tensors are on the same device
#         y_pred, y = y_pred.to(y.device), y.to(y.device)
#         # Custom metric computation
#         score = self.compute_custom_metric(y_pred, y)
#         self.scores.append(score)

#     def compute(self):
#         scores_tensor = torch.tensor(self.scores, device='cuda' if torch.cuda.is_available() else 'cpu')
#         f, not_nans = do_metric_reduction(scores_tensor, self.reduction)
#         return (f, not_nans) if self.get_not_nans else f

#     def aggregate(self, reduction=None):
#         reduction = reduction or self.reduction
#         data = self.get_buffer()
#         if not isinstance(data, torch.Tensor):
#             raise ValueError(f"Expected buffer data to be a PyTorch Tensor, but got {type(data)}.")

#         f, not_nans = do_metric_reduction(data, reduction)
#         return (f, not_nans) if self.get_not_nans else f

#     def compute_custom_metric(self, y_pred, y):
#         # Example custom metric computation logic
#         if self.num_classes is None:
#             self.num_classes = y_pred.shape[1]

#         # must implement - need slice area/volume for muscle classes - 
#         # need this for both ground truth and prediction segmentation masks:
#         # Voxel Volume = Pixel Spacing_x * Pixel Spacing_y * Slice Thickness

#         return 

#     def get_buffer(self):
#         return torch.tensor(self.scores, device='cuda' if torch.cuda.is_available() else 'cpu')



# # Example usage:
# # Instantiate
# custom_metric = CustomMetric(include_background=False, reduction="mean_batch", get_not_nans=False, num_classes=datamodule_cfg['num_classes'])

# # Update with predictions and ground truth
# custom_metric.update(y_pred=combined_mask_onehot, y=local_gt2D_onehot)

# # Aggregate the results
# custom_metric_score = custom_metric.aggregate()




# class CustomMetric(Metric):
#     def __init__(self, include_background=True, reduction='mean'):
#         super().__init__()
#         self.include_background = include_background
#         self.reduction = reduction
#         self.scores = []

#     def reset(self):
#         self.scores = []

#     def update(self, y_pred, y):
#         # Ensure input tensors are on the same device
#         y_pred, y = y_pred.to(y.device), y.to(y.device)
        
#         # Your custom metric computation logic here
#         score = self.compute_custom_metric(y_pred, y)
#         self.scores.append(score)

#     def compute(self):
#         if self.reduction == 'mean':
#             return torch.mean(torch.tensor(self.scores, device='cuda' if torch.cuda.is_available() else 'cpu'))
#         elif self.reduction == 'sum':
#             return torch.sum(torch.tensor(self.scores, device='cuda' if torch.cuda.is_available() else 'cpu'))
#         return torch.tensor(self.scores, device='cuda' if torch.cuda.is_available() else 'cpu')

#     def aggregate(self):
#         # Aggregate scores across all processes if using distributed training
#         if torch.distributed.is_available() and torch.distributed.is_initialized():
#             gathered_scores = evenly_divisible_all_gather(torch.tensor(self.scores, device='cuda' if torch.cuda.is_available() else 'cpu'))
#             self.scores = [score.item() for score in gathered_scores]
        
#         return self.compute()

#     def compute_custom_metric(self, y_pred, y):
#         # Example custom metric computation logic
#         # Replace with your actual metric computation
#         return torch.sum((y_pred == y).float()) / torch.numel(y)
