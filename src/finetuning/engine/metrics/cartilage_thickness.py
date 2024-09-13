
import json
import torch
import numpy as np
from abc import ABC
from torch.nn import Module

# -------------------- DATA POST-PROCESS FUNCTIONS -------------------- #
import numpy as np
from torch.nn import Module
from skimage import transform
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, medial_axis, skeletonize_3d
import matplotlib.pyplot as plt
from skimage import io



class CartilageThicknessMetric(Module):
    def __init__(self, reduction='mean_batch', class_names=None, tissue_labels=None):
        super().__init__()
        self.reduction = reduction
        self.class_names = class_names if class_names else ['Class{}'.format(i) for i in range(1)]
        self.num_classes = len(self.class_names)
        self.tissue_labels = tissue_labels if tissue_labels is not None else list(range(self.num_classes))
        self.cart_thickness_pred = {}
        self.cart_thickness_true = {}

    def reset(self):
        self.cart_thickness_pred = {}
        self.cart_thickness_true = {}

    def resize_mask(self, mask_slice, subject_slice_meta):
        image_size_tuple = [subject_slice_meta['rows'], subject_slice_meta['columns']]
        resized_mask = transform.resize(
            mask_slice,
            image_size_tuple,
            order=0,
            preserve_range=True,
            mode='constant',
            anti_aliasing=False
        )   
        return resized_mask

    def update(self, y_pred, y, subject_slice_meta, subj_id, slice_id):

        # Resize the masks for each class (channel)
        y_pred_resized = np.stack([self.resize_mask(y_pred[0, i, :, :].cpu().numpy(), subject_slice_meta) for i in range(self.num_classes)], axis=0)
        y_resized = np.stack([self.resize_mask(y[0, i, :, :].cpu().numpy(), subject_slice_meta) for i in range(self.num_classes)], axis=0)

        # Add batch dimension back
        y_pred_resized = np.expand_dims(y_pred_resized, axis=0)
        y_resized = np.expand_dims(y_resized, axis=0)

        # Create a unique key for each subject-slice pair
        subject_slice_key = f"{subj_id}_{slice_id}"

        if subject_slice_key not in self.cart_thickness_pred:
            self.cart_thickness_pred[subject_slice_key] = [[] for _ in range(self.num_classes)]
            self.cart_thickness_true[subject_slice_key] = [[] for _ in range(self.num_classes)]

        # Initialize D3d_pred as a 3D array with the same shape as your input, y_pred_resized
        D3d_pred = np.zeros(y_pred_resized.shape[1:])  # Assuming y_pred_resized is (batch_size, num_classes, height, width)
        D3d_true = np.zeros(y_resized.shape[1:])

        for i in range(self.num_classes):
            if i in self.tissue_labels:

                # Apply the medial axis transformation
                skel, distance = medial_axis(y_pred_resized[0, i, :, :].astype(bool), return_distance=True)
                
                # Compute thickness as the distance to the background for points on the skeleton
                thickness_on_skel = distance * skel * subject_slice_meta['slice_thickness']

                # Aggregate thickness (e.g., by averaging)
                mean_thickness = thickness_on_skel[thickness_on_skel > 0].mean()

                # Store the computed thickness values
                self.cart_thickness_pred[subject_slice_key][i].append(mean_thickness)


                # Apply the medial axis transformation
                skel_true, distance_true = medial_axis(y_resized[0, i, :, :].astype(bool), return_distance=True)
                
                # Compute thickness as the distance to the background for points on the skeleton
                thickness_on_skel_true = distance_true * skel_true * subject_slice_meta['slice_thickness']

                # Aggregate thickness (e.g., by averaging)
                mean_thickness_true = thickness_on_skel_true[thickness_on_skel_true > 0].mean()

                # Store the computed thickness values
                self.cart_thickness_true[subject_slice_key][i].append(mean_thickness_true)


    def compute(self):
        # Fill with NaN for labels not in tissue_labels
        for key in self.cart_thickness_pred:
            for i in range(self.num_classes):
                if i not in self.tissue_labels:
                    self.cart_thickness_pred[key][i] = np.nan
                    self.cart_thickness_true[key][i] = np.nan

        return self.cart_thickness_pred, self.cart_thickness_true

    def aggregate_by_subject(self):
        aggregated_cart_thickness_pred = {}
        aggregated_cart_thickness_true = {}

        # Aggregate predicted cartilage thickness
        for key, thicknesses in self.cart_thickness_pred.items():
            subj_id = key[:-4]
            if subj_id not in aggregated_cart_thickness_pred:
                aggregated_cart_thickness_pred[subj_id] = [[] for _ in range(self.num_classes)]
            for i in range(self.num_classes):
                if i in self.tissue_labels:
                    aggregated_cart_thickness_pred[subj_id][i].extend(thicknesses[i])

        # Aggregate true cartilage thickness
        for key, thicknesses in self.cart_thickness_true.items():
            subj_id = key[:-4]
            if subj_id not in aggregated_cart_thickness_true:
                aggregated_cart_thickness_true[subj_id] = [[] for _ in range(self.num_classes)]
            for i in range(self.num_classes):
                if i in self.tissue_labels:
                    aggregated_cart_thickness_true[subj_id][i].extend(thicknesses[i])

        # Compute mean cartilage thickness per subject
        mean_cart_thickness_pred = {
            subj_id: [np.nanmean(aggregated_cart_thickness_pred[subj_id][i]) if len(aggregated_cart_thickness_pred[subj_id][i]) > 0 else np.nan for i in range(self.num_classes)]
            for subj_id in aggregated_cart_thickness_pred
        }

        mean_cart_thickness_true = {
            subj_id: [np.nanmean(aggregated_cart_thickness_true[subj_id][i]) if len(aggregated_cart_thickness_true[subj_id][i]) > 0 else np.nan for i in range(self.num_classes)]
            for subj_id in aggregated_cart_thickness_true
        }

        return mean_cart_thickness_pred, mean_cart_thickness_true

    def forward(self, *args, **kwargs):
        pass