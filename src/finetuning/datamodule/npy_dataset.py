
import os
import random
import numpy as np
from os.path import join
# join = os.path.join

import torch
from torch.utils.data import Dataset

from scipy.ndimage import label as scipy_label
from sklearn.cluster import DBSCAN

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

from .components import build_augmentation_pipeline

# ----------------- Dataset for Finetuning ----------------- #

class mskSAMDataset(Dataset):
    """ Finetuning scripts now in development for instance-based finetuning """
    def __init__(self, root_paths, gt_paths, img_paths, bbox_shift=0, instance_bbox=False, remove_label_ids=[], dataset_name=None, augmentation_config=None):

        self.root_paths = root_paths
        self.gt_path_files = gt_paths
        self.bbox_shift = bbox_shift
        self.instance_bbox = instance_bbox
        self.remove_label_ids = remove_label_ids
        self.datset_name = dataset_name

        # Use the imported function to build the pipeline
        self.augmentation_pipeline = build_augmentation_pipeline(augmentation_config)

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):

        attempt = 0
        max_attempts = len(self.gt_path_files)

        while attempt < max_attempts:
            attempt += 1
                
            # Load npy image (1024, 1024), [0,1]
            img_name = os.path.basename(self.gt_path_files[index])        
            img_1024 = np.load(os.path.join(self.root_paths[index], "imgs", img_name), mmap_mode='r')  # img_1024.shape -> (1024, 1024, 3)

            if img_1024.ndim == 2 or (img_1024.ndim == 3 and img_1024.shape[2] == 1):
                img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)

            # img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)

            # Convert the shape to (3, H, W)
            img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
            
            # Check the minimum and maximum values
            min_val, max_val = img_1024.min(), img_1024.max()

            # If values are outside the range [0, 1], continue to the next iteration
            if min_val < 0 or max_val > 1:
                print(f'Image {img_name} is out of range [{min_val}, {max_val}]. Skipping to the next image.')
                index = (index + 1) % len(self.gt_path_files)
                continue

            # Load npy mask (1024, 1024)
            gt = np.load(self.gt_path_files[index], mmap_mode='r')

            label_ids = np.unique(gt)[1:]  # Exclude background (0)

            # Exclude labels as specified in remove_label_ids
            label_ids = [label for label in label_ids if label not in self.remove_label_ids]

            # Handling the case where all labels are excluded
            if not label_ids:
                raise ValueError(f"All labels have been removed for image at index {index}.")

            chosen_label = random.choice(label_ids)

            gt2D = np.uint8(gt == chosen_label)  # Binary mask for chosen class

            if self.augmentation_pipeline is not None:
                augmented = self.augmentation_pipeline(image=img_1024, mask=gt2D)
                img_1024 = augmented['image']
                gt2D = augmented['mask']

            if self.instance_bbox:
                # Find connected components for the current label
                labeled_array, num_features = scipy_label(gt2D)

                # Randomly choose one of these instances
                chosen_component = random.choice(range(1, num_features + 1))
                component_mask = labeled_array == chosen_component
                
                y_indices, x_indices = np.where(component_mask)

                # Compute the bounding box for the selected component
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # Add perturbation to bounding box coordinates, similar to your original approach
                H, W = gt2D.shape
                x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
                y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

                bboxes = np.array([x_min, y_min, x_max, y_max])
        
                # Update gt2D to only include the selected component
                gt2D = component_mask.astype(np.uint8)
                
            else:
                # Compute bounding box from the augmented single-class mask
                y_indices, x_indices = np.where(gt2D > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                # add perturbation to bounding box coordinates
                H, W = gt2D.shape
                x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
                y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

                bboxes = np.array([x_min, y_min, x_max, y_max])
                
            batch_data = {
                'image': torch.tensor(img_1024).float(),
                'gt2D': torch.tensor(gt2D[None, :, :]).long(),
                'boxes': torch.tensor(bboxes).float(),
                'label_id': torch.tensor(chosen_label).long(),
                'img_name': img_name,
                'dataset_name': self.datset_name,
            }

            return batch_data
        
        # If no valid data found after max_attempts
        raise ValueError("Could not find a valid image after multiple attempts.")
    

# ---------------------------------------------------------- #
# ----------------- Dataset for Evaluation ----------------- #
class MultiClassSAMDataset(Dataset):
    def __init__(self, root_paths, gt_paths, img_paths, bbox_shift=0, mask_labels=None, instance_bbox=False, remove_label_ids=[], use_biomarkers=False, T1rho_map_paths=None, T2_map_paths=None):
        self.root_paths = root_paths
        self.gt_path_files = gt_paths
        self.bbox_shift = bbox_shift
        self.instance_bbox = instance_bbox
        self.remove_label_ids = remove_label_ids
        self.use_biomarkers = use_biomarkers
        self.T1rho_map_paths = T1rho_map_paths
        self.T2_map_paths = T2_map_paths

    def __len__(self):
        return len(self.gt_path_files)
    
    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(join(self.root_paths[index], "imgs", img_name), mmap_mode='r')
        
        if img_1024.ndim == 2 or (img_1024.ndim == 3 and img_1024.shape[2] == 1):
            img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)

        # img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)

        if self.use_biomarkers:
            t1rho = np.load(join(self.root_paths[index], "T1rho_maps", img_name), mmap_mode='r')
            t2 = np.load(join(self.root_paths[index], "T2_maps", img_name), mmap_mode='r')
        
        gt = np.load(self.gt_path_files[index], mmap_mode='r')
        gt = np.array(gt)
        
        label_ids = np.unique(gt)[1:]  # Exclude background
        label_ids = [label for label in label_ids if label not in self.remove_label_ids]

        if not label_ids:
            raise ValueError(f"All labels have been removed for image at index {index}.")

        bbox_list = []
        label_id_list = []

        for label_id in label_ids:
            gt2D = np.uint8(gt == label_id)  # Binary mask for chosen class

            if self.instance_bbox:
                labeled_array, num_features = scipy_label(gt2D)

                for component in range(1, num_features + 1):
                    component_mask = labeled_array == component

                    y_indices, x_indices = np.where(component_mask)

                    # Compute the bounding box for the selected component
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)

                    # No Random perturbation applied to test set
                    H, W = gt2D.shape
                    x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
                    y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

                    bbox = np.array([x_min, y_min, x_max, y_max])

                    bbox_list.append(bbox)
                    label_id_list.append(label_id)
            else:
                y_indices, x_indices = np.where(gt2D > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)

                # Add perturbation to bounding box coordinates
                H, W = gt2D.shape
                x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
                y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

                bbox = np.array([x_min, y_min, x_max, y_max])
                bbox_list.append(bbox)  # Store only bbox array
                label_id_list.append(label_id)  # Corresponding label IDs

        if bbox_list:
            bboxes_tensor = torch.tensor(np.stack(bbox_list)).float()
            label_ids_tensor = torch.tensor(label_id_list).long()
        else:
            bboxes_tensor = torch.tensor([]).float()
            label_ids_tensor = torch.tensor([]).long()

        for label_id in self.remove_label_ids:
            gt[gt == label_id] = 0

        batch_data = {
            'image': torch.tensor(img_1024).float(),
            'gt2D': torch.tensor(gt[None, :, :]).long(),
            'boxes': bboxes_tensor,
            'label_ids': label_ids_tensor,
            'img_name': img_name
        }

        if self.use_biomarkers:
            batch_data.update({
                't1rho': torch.tensor(t1rho).float(),
                't2': torch.tensor(t2).float()
            })

        return batch_data


# ------------------------------------------------------------------------------- #
# ------------------- Dev Components of Evaluation Dataset --------------------- #

class MultiClassNpyDataset(Dataset):
    def __init__(self, root_paths, gt_paths, img_paths, bbox_shift=0, mask_labels=None, instance_bbox=False, remove_label_ids=[]):

        self.root_paths = root_paths
        self.gt_path_files = gt_paths
        self.bbox_shift = bbox_shift
        # self.mask_labels = mask_labels
        self.instance_bbox = instance_bbox
        self.remove_label_ids = remove_label_ids

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])        
        img_1024 = np.load(join(self.root_paths[index], "imgs", img_name), mmap_mode='r')
        
        if img_1024.ndim == 2 or (img_1024.ndim == 3 and img_1024.shape[2] == 1):
            img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)
        # img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)

        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
        
        # load npy mask (1024, 1024)
        gt = np.load(self.gt_path_files[index], mmap_mode='r')

        label_ids = np.unique(gt)[1:]  # Exclude background

        # Exclude labels as specified in remove_label_ids
        label_ids = [label for label in label_ids if label not in self.remove_label_ids]

        # Handling the case where all labels are excluded
        if not label_ids:
            raise ValueError(f"All labels have been removed for image at index {index}.")

        bbox_list = []
        label_id_list = []

        for label_id in label_ids:
            gt2D = np.uint8(gt == label_id) # Binary mask for chosen class

            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
            y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

            bbox = np.array([x_min, y_min, x_max, y_max])
            bbox_list.append(bbox)  # Store only bbox array
            label_id_list.append(label_id)  # Corresponding label IDs

        # Convert list of bboxes to tensor
        if bbox_list:
            bboxes_tensor = torch.tensor(np.stack(bbox_list)).float()
            label_ids_tensor = torch.tensor(label_id_list).long()
        else:
            bboxes_tensor = torch.tensor([]).float()  # Handle case with no bounding boxes
            label_ids_tensor = torch.tensor([]).long()  # Handle case with no label IDs

        for label_id in self.remove_label_ids:
            gt[gt == label_id] = 0

        # Create batch_data dictionary
        batch_data = {
            'image': torch.tensor(img_1024).float(),  
            'gt2D': torch.tensor(gt[None, :, :]).long(),  
            'boxes': bboxes_tensor,
            'label_ids': label_ids_tensor,
            'img_name': img_name,  
            # 'mask_labels': mask_labels
        }

        return batch_data
    

# ------------------------------------------------------------------------------------- #

class MultiInstanceClassNpyDataset(Dataset):
    def __init__(self, root_paths, gt_paths, img_paths, bbox_shift=0, mask_labels=None, instance_bbox=False, remove_label_ids=[]):

        self.root_paths = root_paths
        self.gt_path_files = gt_paths
        self.bbox_shift = bbox_shift
        # self.mask_labels = mask_labels
        self.instance_bbox = instance_bbox
        self.remove_label_ids = remove_label_ids

        self.min_bbox_size = 20 #100

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(join(self.root_paths[index], "imgs", img_name), mmap_mode='r')
        
        if img_1024.ndim == 2 or (img_1024.ndim == 3 and img_1024.shape[2] == 1):
            img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)
        # img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)
        
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
        
        gt = np.load(self.gt_path_files[index], mmap_mode='r')
        gt = np.array(gt)
        label_ids = np.unique(gt)[1:]  # Exclude background

        label_ids = [label for label in label_ids if label not in self.remove_label_ids]

        if not label_ids:
            raise ValueError(f"All labels have been removed for image at index {index}.")

        bbox_list = []
        label_id_list = []

        for label_id in label_ids:
            gt2D = np.uint8(gt == label_id)

            if self.instance_bbox:
                labeled_array, num_features = scipy_label(gt2D)

                for component in range(1, num_features + 1):
                    component_mask = labeled_array == component

                    y_indices, x_indices = np.where(component_mask)

                    # Compute the bounding box for the selected component
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)

                    # No Random perturbation applied to test set
                    H, W = gt2D.shape
                    x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
                    y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

                    bbox = np.array([x_min, y_min, x_max, y_max])

                    bbox_list.append(bbox)
                    label_id_list.append(label_id)

        if bbox_list:
            bboxes_tensor = torch.tensor(np.stack(bbox_list)).float()
            label_ids_tensor = torch.tensor(label_id_list).long()
        else:
            bboxes_tensor = torch.tensor([]).float()
            label_ids_tensor = torch.tensor([]).long()

        for label_id in self.remove_label_ids:
            gt[gt == label_id] = 0

        batch_data = {
            'image': torch.tensor(img_1024).float(),
            'gt2D': torch.tensor(gt[None, :, :]).long(),
            'boxes': bboxes_tensor,
            'label_ids': label_ids_tensor,
            'img_name': img_name
        }

        return batch_data

# ------------------------------------------------------------------------------- #
# ------------------- Dev Components of Finetuning Datasets --------------------- #


class NpyDataset_og(Dataset):
    """ Simplistic, single instance Finetuning Class"""

    def __init__(self, root_paths, gt_paths, img_paths, bbox_shift=0, remove_label_ids=[], augmentation_config=None):

        self.root_paths = root_paths
        self.gt_path_files = gt_paths
        self.bbox_shift = bbox_shift
        self.remove_label_ids = remove_label_ids

        # Use the imported function to build the pipeline
        self.augmentation_pipeline = build_augmentation_pipeline(augmentation_config)

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])        
        img_1024 = np.load(join(self.root_paths[index], "imgs", img_name), mmap_mode='r')
        
        if img_1024.ndim == 2 or (img_1024.ndim == 3 and img_1024.shape[2] == 1):
            img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)
        # img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)

        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
        
        # load npy mask (1024, 1024)
        gt = np.load(self.gt_path_files[index], mmap_mode='r')

        label_ids = np.unique(gt)[1:]  # Exclude background (0)

        # Exclude labels as specified in remove_label_ids
        label_ids = [label for label in label_ids if label not in self.remove_label_ids]

        # Handling the case where all labels are excluded
        if not label_ids:
            raise ValueError(f"All labels have been removed for image at index {index}.")

        chosen_label = random.choice(label_ids.tolist())

        gt2D = np.uint8(gt == chosen_label)  # Binary mask for chosen class

        if self.augmentation_pipeline is not None:
            augmented = self.augmentation_pipeline(image=img_1024, mask=gt2D)
            img_1024 = augmented['image']
            gt2D = augmented['mask']
        
        # Compute bounding box from the augmented single-class mask
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(img_1024.shape[1], x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(img_1024.shape[0], y_max + random.randint(0, self.bbox_shift))

        bboxes = np.array([x_min, y_min, x_max, y_max])

        batch_data = {
            'image': torch.tensor(img_1024).float(),
            'gt2D': torch.tensor(gt2D[None, :, :]).long(),
            'boxes': torch.tensor(bboxes).float(),
            'img_name': img_name,
        }

        return batch_data



# -------------------------------------------------------------------------------------------------

class NpyRandInstanceGTMatchedDataset(Dataset):
    """ Finetuning scripts now in development for instance-based finetuning """
    def __init__(self, root_paths, gt_paths, img_paths, bbox_shift=0, instance_bbox=False, remove_label_ids=[], augmentation_config=None):

        self.root_paths = root_paths
        self.gt_path_files = gt_paths
        self.bbox_shift = bbox_shift
        self.instance_bbox = instance_bbox
        self.remove_label_ids = remove_label_ids
        self.min_bbox_size = 15

        # Use the imported function to build the pipeline
        self.augmentation_pipeline = build_augmentation_pipeline(augmentation_config)

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):

        attempt = 0
        max_attempts = len(self.gt_path_files)

        while attempt < max_attempts:
            attempt += 1
                
            # Load npy image (1024, 1024), [0,1]
            img_name = os.path.basename(self.gt_path_files[index])        
            img_1024 = np.load(os.path.join(self.root_paths[index], "imgs", img_name), mmap_mode='r')
            
            if img_1024.ndim == 2 or (img_1024.ndim == 3 and img_1024.shape[2] == 1):
                img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)
            # img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)  # (1024, 1024, 3)

            # Convert the shape to (3, H, W)
            img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
            
            # Check the minimum and maximum values
            min_val = img_1024.min()
            max_val = img_1024.max()

            # If values are outside the range [0, 1], continue to the next iteration
            if min_val < 0 or max_val > 1:
                print(f'Image {img_name} is out of range [{min_val}, {max_val}]. Skipping to the next image.')
                index = (index + 1) % len(self.gt_path_files)
                continue

            # Load npy mask (1024, 1024)
            gt = np.load(self.gt_path_files[index], mmap_mode='r')

            label_ids = np.unique(gt)[1:]  # Exclude background (0)

            # Exclude labels as specified in remove_label_ids
            label_ids = [label for label in label_ids if label not in self.remove_label_ids]

            # Retry with next index if all labels are excluded
            if not label_ids:
                index = (index + 1) % len(self.gt_path_files)
                continue

            chosen_label = random.choice(label_ids)

            gt2D = np.uint8(gt == chosen_label)  # Binary mask for chosen class

            if self.augmentation_pipeline is not None:
                augmented = self.augmentation_pipeline(image=img_1024, mask=gt2D)
                img_1024 = augmented['image']
                gt2D = augmented['mask']

            if self.instance_bbox:
                # Find connected components for the current label

                labeled_array, num_features = scipy_label(gt2D)

                if num_features > 0:
                    # Randomly choose one of these instances
                    chosen_component = random.choice(range(1, num_features + 1))
                    component_mask = labeled_array == chosen_component
                    
                    y_indices, x_indices = np.where(component_mask)

                    # Compute the bounding box for the selected component
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    
                    # Add perturbation to bounding box coordinates, similar to your original approach
                    H, W = gt2D.shape
                    x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
                    y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

                    bbox = np.array([x_min, y_min, x_max, y_max])

                    # Update gt2D to only include the selected component
                    gt2D = component_mask.astype(np.uint8)
                else:
                    bbox = np.array([0, 0, 0, 0])  # Default value in case no features are found
                    gt2D = np.zeros_like(gt2D, dtype=np.uint8)  # Set gt2D to zeros if no features are found

                bboxes = torch.tensor(bbox).float()
            else:
                # placeholder logic - will implement singular class once design is solidified
                bboxes = torch.zeros((1, 4), dtype=torch.float32)  # Default bbox if instance_bbox is False

            batch_data = {
                'image': torch.tensor(img_1024).float(),
                'gt2D': torch.tensor(gt2D[None, :, :]).long(),
                'boxes': bboxes, 
                'label_id': torch.tensor(chosen_label).long(),
                'img_name': img_name,
            }

            return batch_data
        
        # If no valid data found after max_attempts
        raise ValueError("Could not find a valid image after multiple attempts.")

    


    # ---------------------------------------------------------------------------
    # Biomarker development

    
# class MultiClassSAMDataset(Dataset):
#     def __init__(self, root_paths, gt_paths, img_paths, bbox_shift=0, mask_labels=None, instance_bbox=False, remove_label_ids=[]):
#         self.root_paths = root_paths
#         self.gt_path_files = gt_paths
#         self.bbox_shift = bbox_shift
#         # self.mask_labels = mask_labels
#         self.instance_bbox = instance_bbox
#         self.remove_label_ids = remove_label_ids

#     def __len__(self):
#         return len(self.gt_path_files)
    
#     def __getitem__(self, index):
#         img_name = os.path.basename(self.gt_path_files[index])
#         img_1024 = np.load(join(self.root_paths[index], "imgs", img_name), mmap_mode='r')
#         img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)
#         img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)
        
#         gt = np.load(self.gt_path_files[index], mmap_mode='r')
#         gt = np.array(gt)
        
#         label_ids = np.unique(gt)[1:]  # Exclude background

#         label_ids = [label for label in label_ids if label not in self.remove_label_ids]

#         if not label_ids:
#             raise ValueError(f"All labels have been removed for image at index {index}.")

#         bbox_list = []
#         label_id_list = []

#         for label_id in label_ids:
#             gt2D = np.uint8(gt == label_id) # Binary mask for chosen class

#             if self.instance_bbox:
#                 labeled_array, num_features = scipy_label(gt2D)

#                 for component in range(1, num_features + 1):
#                     component_mask = labeled_array == component

#                     y_indices, x_indices = np.where(component_mask)

#                     # Compute the bounding box for the selected component
#                     x_min, x_max = np.min(x_indices), np.max(x_indices)
#                     y_min, y_max = np.min(y_indices), np.max(y_indices)

#                     # No Random perturbation applied to test set
#                     H, W = gt2D.shape
#                     x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
#                     y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

#                     bbox = np.array([x_min, y_min, x_max, y_max])

#                     bbox_list.append(bbox)
#                     label_id_list.append(label_id)
#             else:
#                 y_indices, x_indices = np.where(gt2D > 0)
#                 x_min, x_max = np.min(x_indices), np.max(x_indices)
#                 y_min, y_max = np.min(y_indices), np.max(y_indices)

#                 # add perturbation to bounding box coordinates
#                 H, W = gt2D.shape
#                 x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
#                 y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

#                 bbox = np.array([x_min, y_min, x_max, y_max])
#                 bbox_list.append(bbox)  # Store only bbox array
#                 label_id_list.append(label_id)  # Corresponding label IDs

#         if bbox_list:
#             bboxes_tensor = torch.tensor(np.stack(bbox_list)).float()
#             label_ids_tensor = torch.tensor(label_id_list).long()
#         else:
#             bboxes_tensor = torch.tensor([]).float()
#             label_ids_tensor = torch.tensor([]).long()

#         for label_id in self.remove_label_ids:
#             gt[gt == label_id] = 0

#         batch_data = {
#             'image': torch.tensor(img_1024).float(),
#             'gt2D': torch.tensor(gt[None, :, :]).long(),
#             'boxes': bboxes_tensor,
#             'label_ids': label_ids_tensor,
#             'img_name': img_name
#         }

#         return batch_data

# ------------------------------------------------------------------------- #
# ----------------- Dataset for Maps Biomarker Evaluation ----------------- #

# class MultiClassBiomarkerSAMDataset(Dataset):
#     def __init__(self, root_paths, gt_paths, img_paths, T1rho_map_paths, T2_map_paths, bbox_shift=0, mask_labels=None, instance_bbox=False, remove_label_ids=[]):
#         self.root_paths = root_paths
#         self.gt_path_files = gt_paths

#         self.T1rho_map_paths = T1rho_map_paths
#         self.T2_map_paths = T2_map_paths

#         self.bbox_shift = bbox_shift
#         # self.mask_labels = mask_labels
#         self.instance_bbox = instance_bbox
#         self.remove_label_ids = remove_label_ids

#     def __len__(self):
#         return len(self.gt_path_files)
    
#     def __getitem__(self, index):
#         img_name = os.path.basename(self.gt_path_files[index])
#         img_1024 = np.load(join(self.root_paths[index], "imgs", img_name), mmap_mode='r')
#         img_1024 = np.repeat(img_1024[:, :, None], 3, axis=-1)
#         img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, 1024, 1024)

#         # import pdb; pdb.set_trace()

#         t1rho = np.load(join(self.root_paths[index], "T1rho_maps", img_name), mmap_mode='r')
#         t2 = np.load(join(self.root_paths[index], "T2_maps", img_name), mmap_mode='r')
        
#         gt = np.load(self.gt_path_files[index], mmap_mode='r')
#         gt = np.array(gt)
        
#         label_ids = np.unique(gt)[1:]  # Exclude background

#         label_ids = [label for label in label_ids if label not in self.remove_label_ids]

#         if not label_ids:
#             raise ValueError(f"All labels have been removed for image at index {index}.")

#         bbox_list = []
#         label_id_list = []

#         for label_id in label_ids:
#             gt2D = np.uint8(gt == label_id) # Binary mask for chosen class

#             if self.instance_bbox:
#                 labeled_array, num_features = scipy_label(gt2D)

#                 for component in range(1, num_features + 1):
#                     component_mask = labeled_array == component

#                     y_indices, x_indices = np.where(component_mask)

#                     # Compute the bounding box for the selected component
#                     x_min, x_max = np.min(x_indices), np.max(x_indices)
#                     y_min, y_max = np.min(y_indices), np.max(y_indices)

#                     # No Random perturbation applied to test set
#                     H, W = gt2D.shape
#                     x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
#                     y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

#                     bbox = np.array([x_min, y_min, x_max, y_max])

#                     bbox_list.append(bbox)
#                     label_id_list.append(label_id)
#             else:
#                 y_indices, x_indices = np.where(gt2D > 0)
#                 x_min, x_max = np.min(x_indices), np.max(x_indices)
#                 y_min, y_max = np.min(y_indices), np.max(y_indices)

#                 # add perturbation to bounding box coordinates
#                 H, W = gt2D.shape
#                 x_min, x_max = max(0, x_min - random.randint(0, self.bbox_shift)), min(W, x_max + random.randint(0, self.bbox_shift))
#                 y_min, y_max = max(0, y_min - random.randint(0, self.bbox_shift)), min(H, y_max + random.randint(0, self.bbox_shift))

#                 bbox = np.array([x_min, y_min, x_max, y_max])
#                 bbox_list.append(bbox)  # Store only bbox array
#                 label_id_list.append(label_id)  # Corresponding label IDs

#         if bbox_list:
#             bboxes_tensor = torch.tensor(np.stack(bbox_list)).float()
#             label_ids_tensor = torch.tensor(label_id_list).long()
#         else:
#             bboxes_tensor = torch.tensor([]).float()
#             label_ids_tensor = torch.tensor([]).long()

#         for label_id in self.remove_label_ids:
#             gt[gt == label_id] = 0


#         # import pdb; pdb.set_trace()

#         batch_data = {
#             'image': torch.tensor(img_1024).float(),
#             'gt2D': torch.tensor(gt[None, :, :]).long(),
#             'boxes': bboxes_tensor,
#             'label_ids': label_ids_tensor,
#             'img_name': img_name,
#             't1rho': torch.tensor(t1rho).float(),
#             't2': torch.tensor(t2).float()
#         }

#         return batch_data

