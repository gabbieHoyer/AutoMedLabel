
import json
import torch
import numpy as np
from abc import ABC
from torch.nn import Module


# -------------- Base Maps Class -------------- #
class MapsMetric(Module, ABC):
    def __init__(self, map_type, reduction='mean_batch', class_names=None, tissue_labels=None):
        super().__init__()
        self.map_type = map_type
        self.reduction = reduction
        self.class_names = class_names if class_names else ['Class{}'.format(i) for i in range(1)]
        self.num_classes = len(self.class_names)
        self.tissue_labels = tissue_labels if tissue_labels is not None else list(range(self.num_classes))
        self.map_values_pred = {}
        self.map_values_true = {}

    def reset(self):
        self.map_values_pred = {}
        self.map_values_true = {}

    def update(self, y_pred, y, map_array, subject_slice_meta, subj_id, slice_id):
        # Create a unique key for each subject-slice pair
        subject_slice_key = f"{subj_id}_{slice_id}"

        if subject_slice_key not in self.map_values_pred:
            self.map_values_pred[subject_slice_key] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class
            self.map_values_true[subject_slice_key] = [0.0] * (self.num_classes + 1)  # +1 for the 'total' class

        seg_vol_pred = y_pred.cpu().numpy()
        seg_vol_true = y.cpu().numpy()
        map_vol = map_array.cpu().numpy()

        # Clip map values
        map_vol = np.clip(map_vol, 0, 100)

        total_map_pred = 0.0
        total_map_true = 0.0

        for i in range(self.num_classes):
            binary_seg_pred = seg_vol_pred[:, i, :, :]
            binary_seg_true = seg_vol_true[:, i, :, :]

            # Compute the average map value for each class in this slice
            map_value_pred = (map_vol * binary_seg_pred).sum() / binary_seg_pred.sum() if binary_seg_pred.sum() != 0 else np.nan
            map_value_true = (map_vol * binary_seg_true).sum() / binary_seg_true.sum() if binary_seg_true.sum() != 0 else np.nan

            self.map_values_pred[subject_slice_key][i] = map_value_pred
            self.map_values_true[subject_slice_key][i] = map_value_true

            if i in self.tissue_labels:
                total_map_pred += map_value_pred if not np.isnan(map_value_pred) else 0.0
                total_map_true += map_value_true if not np.isnan(map_value_true) else 0.0

        # Update the 'total' class average for this slice
        valid_pred_count = len([v for v in self.map_values_pred[subject_slice_key][:-1] if not np.isnan(v)])
        valid_true_count = len([v for v in self.map_values_true[subject_slice_key][:-1] if not np.isnan(v)])
        self.map_values_pred[subject_slice_key][-1] = total_map_pred / valid_pred_count if valid_pred_count > 0 else np.nan
        self.map_values_true[subject_slice_key][-1] = total_map_true / valid_true_count if valid_true_count > 0 else np.nan

    def compute(self):
        # Fill with NaN for labels not in tissue_labels
        for key in self.map_values_pred:
            for i in range(self.num_classes):
                if i not in self.tissue_labels:
                    self.map_values_pred[key][i] = np.nan
                    self.map_values_true[key][i] = np.nan

        return self.map_values_pred, self.map_values_true

    def aggregate_by_subject(self):
        aggregated_map_values_pred = {}
        aggregated_map_values_true = {}
        count_non_zero_pred = {}
        count_non_zero_true = {}

        for key, map_values in self.map_values_pred.items():
            subj_id = key[:-4]  # Extract subject ID
            if subj_id not in aggregated_map_values_pred:
                aggregated_map_values_pred[subj_id] = [0.0] * (self.num_classes + 1)
                count_non_zero_pred[subj_id] = [0] * (self.num_classes + 1)

            for i in range(self.num_classes):
                if i in self.tissue_labels:
                    if not np.isnan(map_values[i]):
                        aggregated_map_values_pred[subj_id][i] += map_values[i]
                        count_non_zero_pred[subj_id][i] += 1
            if not np.isnan(map_values[-1]):
                aggregated_map_values_pred[subj_id][-1] += map_values[-1]
                count_non_zero_pred[subj_id][-1] += 1

        for key, map_values in self.map_values_true.items():
            subj_id = key[:-4] # Extract subject ID
            if subj_id not in aggregated_map_values_true:
                aggregated_map_values_true[subj_id] = [0.0] * (self.num_classes + 1)
                count_non_zero_true[subj_id] = [0] * (self.num_classes + 1)

            for i in range(self.num_classes):
                if i in self.tissue_labels:
                    if not np.isnan(map_values[i]):
                        aggregated_map_values_true[subj_id][i] += map_values[i]
                        count_non_zero_true[subj_id][i] += 1
            if not np.isnan(map_values[-1]):
                aggregated_map_values_true[subj_id][-1] += map_values[-1]
                count_non_zero_true[subj_id][-1] += 1

        # Average the values by the number of non-zero slices per subject
        for subj_id, values in aggregated_map_values_pred.items():
            for i in range(self.num_classes):
                if i in self.tissue_labels:
                    count = count_non_zero_pred[subj_id][i]
                    if count > 0:
                        aggregated_map_values_pred[subj_id][i] /= count
            count_total = count_non_zero_pred[subj_id][-1]
            if count_total > 0:
                aggregated_map_values_pred[subj_id][-1] /= count_total

        for subj_id, values in aggregated_map_values_true.items():
            for i in range(self.num_classes):
                if i in self.tissue_labels:
                    count = count_non_zero_true[subj_id][i]
                    if count > 0:
                        aggregated_map_values_true[subj_id][i] /= count
            count_total = count_non_zero_true[subj_id][-1]
            if count_total > 0:
                aggregated_map_values_true[subj_id][-1] /= count_total

        return aggregated_map_values_pred, aggregated_map_values_true

    def forward(self, *args, **kwargs):
        pass



# ----------- Derived Classes --------------- #

class T1rhoMetric(MapsMetric):
    def __init__(self, reduction='mean_batch', class_names=None, tissue_labels=None):
        super().__init__(map_type='T1rho', reduction=reduction, class_names=class_names, tissue_labels=tissue_labels)

class T2Metric(MapsMetric):
    def __init__(self, reduction='mean_batch', class_names=None, tissue_labels=None):
        super().__init__(map_type='T2', reduction=reduction, class_names=class_names, tissue_labels=tissue_labels)

