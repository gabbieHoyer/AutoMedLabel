
import json
import torch
import numpy as np
from abc import ABC
from torch.nn import Module

import cv2
import math
from scipy.spatial import ConvexHull, QhullError
from scipy.ndimage import center_of_mass, label as scipy_label

import logging
from skimage import transform


# -------------------------------------------------------------------------------------------- #

class TissueHeightMetric(Module):
    def __init__(self, reduction='mean_batch', class_names=None, tissue_labels=None):
        super().__init__()
        self.reduction = reduction
        self.class_names = class_names if class_names else ['Class{}'.format(i) for i in range(1)]
        self.num_classes = len(self.class_names)
        self.tissue_labels = tissue_labels if tissue_labels is not None else list(range(self.num_classes))
        self.heights_pred = {}
        self.heights_true = {}

    def reset(self):
        self.heights_pred = {}
        self.heights_true = {}

    def resize_mask(self, mask_slice, subject_slice_meta):
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
        subject_slice_key = f"{subj_id}_{slice_id}"

        if subject_slice_key not in self.heights_pred:
            self.heights_pred[subject_slice_key] = {}
            self.heights_true[subject_slice_key] = {}

        # Resize the masks for each class (channel)
        y_pred_resized = np.stack([self.resize_mask(y_pred[0, i, :, :].cpu().numpy(), subject_slice_meta) for i in range(self.num_classes)], axis=0)
        y_resized = np.stack([self.resize_mask(y[0, i, :, :].cpu().numpy(), subject_slice_meta) for i in range(self.num_classes)], axis=0)

        # Add batch dimension back
        y_pred_resized = np.expand_dims(y_pred_resized, axis=0)
        y_resized = np.expand_dims(y_resized, axis=0)

        for i in range(self.num_classes):
            class_name = self.class_names[i]
            key = f"{class_name}"
            if i == 0:
                self.heights_pred[subject_slice_key][key] = np.nan
                self.heights_true[subject_slice_key][key] = np.nan
                continue  # Skip background label

            if i in self.tissue_labels:
                # Directly use the mask without instance separation
                mask_pred = y_pred_resized[0, i, :, :]
                mask_true = y_resized[0, i, :, :]

                height_pred = self.compute_height(mask_pred, subject_slice_meta['pixel_spacing'])
                height_true = self.compute_height(mask_true, subject_slice_meta['pixel_spacing'])

                self.heights_pred[subject_slice_key][key] = height_pred
                self.heights_true[subject_slice_key][key] = height_true

    def compute_height(self, segmentation, pixel_spacing):
        coords = self.min_bounding_rectangle(segmentation)
        height = self.extract_height(coords, pixel_spacing)
        return height

    def euclidean_dist(self, x1, x2, y1, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def min_bounding_rectangle(self, mask):
        pi2 = np.pi / 2.

        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0 or len(contours[0]) < 3:
            return np.zeros((4, 2))  # Return a default rectangle if not enough points

        points = np.concatenate(contours).reshape(-1, 2)

        if np.all(points == 0):
            return np.zeros((4, 2))

        try:
            hull_points = points[ConvexHull(points).vertices]
        except QhullError:
            return np.zeros((4, 2))

        edges = np.diff(hull_points, axis=0)
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)

        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles - pi2),
            np.cos(angles + pi2),
            np.cos(angles)]).T.reshape((-1, 2, 2))

        rot_points = np.dot(rotations, hull_points.T)

        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        areas = (max_x - min_x) * (max_y - min_y)

        best_idx = np.argmin(areas)

        x1, x2 = max_x[best_idx], min_x[best_idx]
        y1, y2 = max_y[best_idx], min_y[best_idx]

        r = rotations[best_idx]

        rval = np.around(np.dot(np.array([[x1, y2], [x2, y2], [x2, y1], [x1, y1]]), r), 3)
        return rval

    def extract_height(self, coords, pixel_spacing):
        distances = [
            self.euclidean_dist(coords[0][0], coords[1][0], coords[0][1], coords[1][1]),
            self.euclidean_dist(coords[0][0], coords[2][0], coords[0][1], coords[2][1]),
            self.euclidean_dist(coords[0][0], coords[3][0], coords[0][1], coords[3][1])
        ]

        min_dist = min(distances)
        idx = distances.index(min_dist)

        x_dist = coords[0][0] - coords[idx+1][0]
        y_dist = coords[0][1] - coords[idx+1][1]
        x_dist *= pixel_spacing[1]
        y_dist *= pixel_spacing[0]

        disc_height = math.sqrt(x_dist**2 + y_dist**2)
        return disc_height

    def compute(self):
        all_keys = set()
        # Collect all possible class keys
        for heights in self.heights_pred.values():
            all_keys.update(heights.keys())

        # Ensure all keys are present for each slice
        for key in self.heights_pred:
            for class_key in all_keys:
                if class_key not in self.heights_pred[key]:
                    self.heights_pred[key][class_key] = np.nan
                if class_key not in self.heights_true[key]:
                    self.heights_true[key][class_key] = np.nan
        return self.heights_pred, self.heights_true

    def aggregate_by_subject(self):
        all_keys = set()
        for heights in self.heights_pred.values():
            all_keys.update(heights.keys())

        aggregated_heights_pred = {}
        aggregated_heights_true = {}

        for key, heights in self.heights_pred.items():
            subj_id = key[:-4]

            if subj_id not in aggregated_heights_pred:
                # Initialize all keys with NaN
                aggregated_heights_pred[subj_id] = {k: np.nan for k in all_keys}

            for class_key in all_keys:
                current_height = heights.get(class_key, np.nan)
                existing_height = aggregated_heights_pred[subj_id].get(class_key, np.nan)
                aggregated_heights_pred[subj_id][class_key] = np.nanmax([existing_height, current_height])

        # Repeat the same for heights_true
        for key, heights in self.heights_true.items():
            subj_id = key[:-4]

            if subj_id not in aggregated_heights_true:
                aggregated_heights_true[subj_id] = {k: np.nan for k in all_keys}

            for class_key in all_keys:
                current_height = heights.get(class_key, np.nan)
                existing_height = aggregated_heights_true[subj_id].get(class_key, np.nan)
                aggregated_heights_true[subj_id][class_key] = np.nanmax([existing_height, current_height])

        return aggregated_heights_pred, aggregated_heights_true

    def forward(self, *args, **kwargs):
        pass

