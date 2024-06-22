from monai.metrics import DiceMetric, HausdorffDistanceMetric
import numpy as np

import torch
from monai.transforms import AsDiscrete
from monai.metrics import compute_iou


def to_one_hot(y, num_classes):
    """Convert masks to one-hot encoding."""
    # Assuming y is a tensor of shape [num_slices, H, W]
    # Convert to [B, num_slices, H, W] where B=1 for batch dim
    y = y.unsqueeze(0)
    # Convert to one-hot; shape becomes [B, num_classes, num_slices, H, W]
    return AsDiscrete(to_onehot=num_classes)(y)

def compute_metric_per_label(metric_function, mask_gt, mask_pred, labels=None):
    """Compute a given metric for each label in multi-label segmentation masks."""
    if labels is None:
        labels = np.unique(np.concatenate([np.unique(mask_gt), np.unique(mask_pred)]))
    metric_scores = {}
    for label in labels:
        if label == 0:  # Optionally skip background
            continue
        mask_gt_label = (mask_gt == label)
        mask_pred_label = (mask_pred == label)
        metric_score = metric_function(mask_gt_label, mask_pred_label)
        metric_scores[label] = metric_score
    return metric_scores

# --------------------------------------------------------------
def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute Sørensen-Dice coefficient."""
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

def compute_multilabel_dice_coefficients(mask_gt, mask_pred):
    """Compute Dice coefficient for each unique label in the masks."""
    dice_scores = {}
    labels = np.unique(np.concatenate([np.unique(mask_gt), np.unique(mask_pred)]))
    for label in labels:
        if label == 0:  # Skip background
            continue
        mask_gt_label = mask_gt == label
        mask_pred_label = mask_pred == label
        volume_sum = mask_gt_label.sum() + mask_pred_label.sum()
        if volume_sum == 0:
            dice_scores[label] = np.NaN
        else:
            volume_intersect = (mask_gt_label & mask_pred_label).sum()
            dice_scores[label] = 2 * volume_intersect / volume_sum
    return dice_scores

# --------------------------------------------------------------

def compute_IoU(mask_gt, mask_pred, num_classes):
    """Compute IoU for each label in multi-label segmentation masks."""
    mask_gt_tensor = torch.tensor(mask_gt, dtype=torch.int64)
    mask_pred_tensor = torch.tensor(mask_pred, dtype=torch.int64)

    mask_gt_one_hot = to_one_hot(mask_gt_tensor, num_classes)
    mask_pred_one_hot = to_one_hot(mask_pred_tensor, num_classes)

    iou_scores = compute_iou(y_pred=mask_pred_one_hot, y=mask_gt_one_hot, include_background=False, ignore_empty=False)

    # Here we take the mean of the iou scores tensor for each class and convert to scalar
    iou_score_dict = {label: torch.mean(scores).item() for label, scores in enumerate(iou_scores, start=1)}

    return iou_score_dict

# --------------------------------------------------------------
def compute_hausdorff_distance(mask_gt, mask_pred):
    """Compute the Hausdorff distance for 3D volume masks."""
    # Initialize the Hausdorff distance metric
    hausdorff_distance_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    
    # Ensure mask_gt and mask_pred are PyTorch tensors
    mask_gt_tensor = torch.tensor(mask_gt, dtype=torch.float32)
    mask_pred_tensor = torch.tensor(mask_pred, dtype=torch.float32)
    
    # Add a batch and channel dimension: [D, H, W] -> [B, C, D, H, W] for compatibility with MONAI
    # Here, B=1 for batch size and C=1 for the number of channels
    mask_gt_tensor = mask_gt_tensor.unsqueeze(0).unsqueeze(0)
    mask_pred_tensor = mask_pred_tensor.unsqueeze(0).unsqueeze(0)
    
    # Compute the Hausdorff distance
    distance = hausdorff_distance_metric(y_pred=mask_pred_tensor, y=mask_gt_tensor)
    
    # Convert to a scalar value if needed
    return distance.item()

# --------------------------------------------------------------
def compute_multiclass_sensitivity_specificity(mask_gt, mask_pred):
    sens_results = []
    spec_results = []
    
    for class_value in np.unique(mask_gt):
        if class_value == 0:  # Optionally skip background
            continue
        
        binary_mask_gt = (mask_gt == class_value)
        binary_mask_pred = (mask_pred == class_value)
        
        TP = np.sum((binary_mask_gt == 1) & (binary_mask_pred == 1))
        FN = np.sum((binary_mask_gt == 1) & (binary_mask_pred == 0))
        TN = np.sum((binary_mask_gt == 0) & (binary_mask_pred == 0))
        FP = np.sum((binary_mask_gt == 0) & (binary_mask_pred == 1))
        
        sensitivity = TP / (TP + FN) if TP + FN > 0 else np.NaN
        specificity = TN / (TN + FP) if TN + FP > 0 else np.NaN
        
        sens_results.append(f"Label {class_value}: {sensitivity:.3f}")
        spec_results.append(f"Label {class_value}: {specificity:.3f}")
    
    # Joining all results into single strings
    sens_str = "; ".join(sens_results)
    spec_str = "; ".join(spec_results)
    
    return sens_str, spec_str


