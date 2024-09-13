
import json
import torch
import numpy as np
from abc import ABC
from torch.nn import Module
from skimage import transform

# -------------------------------------------------- #
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


