import numpy as np
import random
# For instance based bboxes
from sklearn.cluster import DBSCAN
from scipy.ndimage import label as scipy_label

def identify_instance_bbox_from_slice(gt, bbox_shift):
    # User defined? From GT?

    label_ids = np.unique(gt[gt!=0])  # Exclude background

    bbox_list = []
    for label_id in label_ids:
        gt_label_id_resized = np.uint8(gt == label_id)

        # Find connected components for the current label
        labeled_array, num_features = scipy_label(gt_label_id_resized)
        
        for component in range(1, num_features + 1):
            component_mask = labeled_array == component
            y_indices, x_indices = np.where(component_mask)

            # Use DBSCAN or another clustering method if necessary
            # For simplicity, we'll use the centroids here. Adjust based on your needs.
            if len(y_indices) == 0 and len(x_indices) == 0:
                continue

            X = np.column_stack([x_indices, y_indices])
            clustering = DBSCAN(eps=50, min_samples=2).fit(X)  # Adjust eps based on your domain knowledge
            unique_clusters = np.unique(clustering.labels_)
            
            for cls_label in unique_clusters:  # Use a different variable name here to avoid conflict
                if cls_label == -1:  # Ignore noise
                    continue
                cluster_mask = clustering.labels_ == cls_label
                cluster_x_indices = x_indices[cluster_mask]
                cluster_y_indices = y_indices[cluster_mask]
                    
                # Compute and store the bounding box for this cluster
                x_min, x_max = np.min(cluster_x_indices), np.max(cluster_x_indices)
                y_min, y_max = np.min(cluster_y_indices), np.max(cluster_y_indices)
                
                # Add perturbation to bounding box coordinates, similar to your original approach
                H, W = gt_label_id_resized.shape
                x_min, x_max = max(0, x_min - random.randint(0, bbox_shift)), min(W, x_max + random.randint(0, bbox_shift))
                y_min, y_max = max(0, y_min - random.randint(0, bbox_shift)), min(H, y_max + random.randint(0, bbox_shift))
                
                bbox = np.array([x_min, y_min, x_max, y_max])
                bbox_list.append((label_id, bbox))  # meant for visualization

    return bbox_list

def identify_instance_bbox_from_volume(gt_3D, bbox_shift):
    """
    Params:
    - gt_3d: 3d image with dims slice x height x width
    Returns:
    -  a list for each slice that has a list of tuples for (label_id, bboxes)
    """
    volume_bbox_list = []
    for slice in range(gt_3D.shape[0]):
        gt = gt_3D[slice]
        slice_bbox_list = identify_instance_bbox_from_slice(gt, bbox_shift)
        if slice_bbox_list:
            volume_bbox_list.append((slice, slice_bbox_list))
    return volume_bbox_list

def identify_bbox_from_slice(gt, bbox_shift):
    """
    Params:
    - gt: 2d image with dims 1024x1024
    Returns:
    -  a list of tuples for (label_id, bboxes)
    """
    # User defined? From GT?

    label_ids = np.unique(gt[gt!=0])  # Exclude background

    bbox_list = []
    for label_id in label_ids:
        gt_label_id_resized = np.uint8(gt == label_id)
        y_indices, x_indices = np.where(gt_label_id_resized > 0)
        
        if y_indices.size == 0 or x_indices.size == 0:  # Skip if no label found in resized GT
            continue

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = gt_label_id_resized.shape
        x_min, x_max = max(0, x_min - random.randint(0, bbox_shift)), min(W, x_max + random.randint(0, bbox_shift))
        y_min, y_max = max(0, y_min - random.randint(0, bbox_shift)), min(H, y_max + random.randint(0, bbox_shift))

        bbox = np.array([x_min, y_min, x_max, y_max])
        bbox_list.append((label_id, bbox))
    return bbox_list

def identify_bbox_from_volume(gt_3D, bbox_shift):
    """
    Params:
    - gt_3d: 3d image with dims slice x height x width
    Returns:
    -  a list for each slice that has a list of tuples for (label_id, bboxes)
    """
    volume_bbox_list = []

    for slice in range(gt_3D.shape[0]):
        gt = gt_3D[slice]
        slice_bbox_list = identify_bbox_from_slice(gt, bbox_shift)
        if slice_bbox_list:
            volume_bbox_list.append((slice, slice_bbox_list))
    return volume_bbox_list

def adjust_bbox_to_new_img_size(bbox, current_img_with_bbox_size, new_img_with_bbox_size):
    """
    Adjust bounding box coordinates from resized image size back to original image size.

    Parameters:
    - bbox: The bounding box coordinates as (x_min, y_min, x_max, y_max).
    - current_img_with_bbox_size: The shape (height, width) of the original image.
    - resized_shape: The shape (height, width) of the resized image.

    Returns:
    - Adjusted bounding box coordinates as (x_min, y_min, x_max, y_max).
    """
    # Calculate scaling factors
    y_scale = new_img_with_bbox_size[0] / current_img_with_bbox_size[0]
    x_scale = new_img_with_bbox_size[1] / current_img_with_bbox_size[1]

    # Apply scaling factors to the bounding box coordinates
    x_min, y_min, x_max, y_max = bbox
    adjusted_bbox = [x_min * x_scale, y_min * y_scale, x_max * x_scale, y_max * y_scale]

    return np.array(adjusted_bbox)

def update_vol_bbox_info_to_new_slices(volume_bbox_list, selected_slices):
    updated_volume_bbox_list = []
    volume_bbox_dict = dict(volume_bbox_list)
    for new_slice_idx in range(len(selected_slices)):
        old_slice_idx = selected_slices[new_slice_idx]
        if old_slice_idx in volume_bbox_dict.keys():
            updated_volume_bbox_list.append((new_slice_idx, volume_bbox_dict[old_slice_idx]))
    
    return updated_volume_bbox_list
