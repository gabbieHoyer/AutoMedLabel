
from typing import Union #, Tuple, List
import numpy as np

from scipy.ndimage import center_of_mass
from scipy.ndimage import label as scipy_label
import cc3d
from skimage import transform

# ---------------------- UTILITY FUNCTIONS ----------------------

def find_non_zero_slices(mask_data: np.ndarray) -> np.ndarray:
    """Identify indices of slices that contain non-zero values."""
    z_indices = np.unique(np.where(mask_data > 0)[0])
    return z_indices

def crop_data_to_non_zero_slices(data: np.ndarray, z_indices: np.ndarray) -> np.ndarray:
    """Crop the 3D data to only include slices with non-zero values."""
    return data[z_indices]

# ------------------ DATA TRANSFORMATION FUNCTIONS ------------------

class MaskPrep():
    """
    Class of static variables and functions to prepare masks for SAM. This class may handle ground 
    truths for comparison, or mask prompts. Supports 2D or 3D mask input.
    
    Static Params:
    - remove_label_ids: A list of label IDs to be removed from the mask data.
    - target_label_id: 
    - voxel_threshold_3d: The voxel count threshold for removing small objects in 3D.
    - pixel_threshold_2d: The pixel count threshold for removing small objects in 2D slices.
    - image_size_tuple: (H, W) in pixels to resize mask using nearest-neighbor interpolation. This 
        will preserve image intensities.
    - crop_non_zero_slices_flag: True or False. Indicate whether to crop volume to only slices with 
        segmentations if the mask is 3D.
    
    Input Params:
    - mask_data: The mask data as a numpy array. Supports 2D slices or 3D volumes.
    """
    def __init__(self, remove_label_ids: list = [], target_label_id: Union[int, list[int]] = [], voxel_threshold_3d: int = 0, pixel_threshold_2d: int = 0, image_size_tuple:tuple[int, int] = [], crop_non_zero_slices_flag:bool=True, make_square:bool=False, ratio_resize:bool=False):
        self.remove_label_ids = remove_label_ids
        self.target_label_id = target_label_id
        self.voxel_threshold_3d = voxel_threshold_3d
        self.pixel_threshold_2d = pixel_threshold_2d
        self.image_size_tuple = image_size_tuple
        self.crop_non_zero_slices_flag = crop_non_zero_slices_flag
        self.make_square = make_square
        self.ratio_resize = ratio_resize

    # -------------------- UTILITY FUNCTIONS --------------------
        
    def remove_labels(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Remove specified labels from the mask data.
        Supports 2D and 3D.

        Parameters:
        - mask_data: The mask data as a numpy array.
        - label_ids: A list of label IDs to be removed from the mask data.

        Returns:
        - A numpy array with the specified labels removed.
        """
        remove_label_ids = self.remove_label_ids
        
        # Further check if label_ids list is empty; if so, return the original mask_data without changes
        if not remove_label_ids:
            return mask_data
        # Validate that label_ids is a list
        if not isinstance(remove_label_ids, list):
            raise TypeError("label_ids must be a list.")
        # Validate that mask_data is a numpy array
        if not isinstance(mask_data, np.ndarray):
            raise TypeError("mask_data must be a numpy array.")
        
        
        # Proceed to remove specified labels
        mask_data[np.isin(mask_data, remove_label_ids)] = 0
        return mask_data

    def label_instances(self, mask_data: np.ndarray) -> tuple[np.ndarray, int]:
        """
        dims may be '2D' or '3D'
        """
        target_label_id = self.target_label_id

        # Check if label_ids list is empty; if so, return the original mask_data without changes
        if not target_label_id:
            return mask_data, None
        # Validate that label_ids is a list
        if isinstance(target_label_id, int):
            target_label_id = [target_label_id]  # Convert to list if only one ID is provided
        
        total_num_instances = 0
        max_existing_label = np.max(mask_data)

        if len(np.shape(mask_data)) == 3:
            cc3d_connectivity = 26
        elif len(np.shape(mask_data)) == 2:
            cc3d_connectivity = 8

        for label_id in target_label_id:
            if label_id not in mask_data:
                raise UserWarning("No non-zero slices for volume.")
                continue  # Skip if current ID is not in mask_data
            
            # Find connected components
            target_binary = mask_data == label_id
            mask_data[target_binary] = 0
            labeled_mask, num_labels = cc3d.connected_components(target_binary, connectivity=cc3d_connectivity, return_N=True)
            
            # Calculate centroids for each label and sort them
            centroids = [center_of_mass(target_binary, labeled_mask, i) for i in range(1, num_labels + 1)]
            
            # Debug: print centroids to check if they make sense
            print("Centroids before sorting:", centroids)
            
            # Assuming the second index [1] corresponds to the superior-inferior axis (change if necessary)
            # Change the sorting order if needed, depending on the physical orientation of your volume
            sorted_indices = np.argsort([c[2] for c in centroids])  # c[2] ok, c[1] bad for bacpac sag
            
            # Debug: print sorted indices to check the order
            print("Sorted indices:", sorted_indices)
                    
            # Debug: print new labels to check the mapping
            print("New labels mapping:", [(idx + 1, rank + max_existing_label) for rank, idx in enumerate(sorted_indices)])

            # Map sorted indices to new labels
            new_labels = np.zeros_like(labeled_mask)
            for rank, idx in enumerate(sorted_indices, start=1):
                new_labels[labeled_mask == idx + 1] = rank + max_existing_label
                print(f"Mapping original label {idx + 1} to new label {rank + max_existing_label}")

            # Update mask_data with new labels
            mask_data[new_labels > 0] = new_labels[new_labels > 0]
            
            # Update the max label and total instances count
            max_existing_label = np.max(mask_data)
            total_num_instances += num_labels
        
        return mask_data, total_num_instances

    def remove_small_objects(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Apply small object removal in both 3D and 2D to the mask data.
        Removes small objects based on a voxel count threshold for 3D and a pixel count threshold for 2D slices.
        
        Parameters:
        - mask_data: The mask data as a numpy array.
        - voxel_threshold_3d: The voxel count threshold for removing small objects in 3D.
        - pixel_threshold_2d: The pixel count threshold for removing small objects in 2D slices.
        
        Returns:
        - A numpy array with small objects removed.
        """
        def remove_small_objects_2D(mask_data):
            mask_data = cc3d.dust(mask_data, threshold=self.pixel_threshold_2d, connectivity=8, in_place=True)
            return mask_data

        dims = len(np.shape(mask_data))
        if dims == 2:
            mask_data = remove_small_objects_2D(mask_data)
        elif dims == 3:
            # Remove small objects in 3D
            mask_data = cc3d.dust(mask_data, threshold=self.voxel_threshold_3d, connectivity=26, in_place=True)
            # Iteratively remove small objects from each 2D slice
            for slice_index in range(mask_data.shape[0]):
                mask_data[slice_index, :, :] = remove_small_objects_2D(mask_data[slice_index, :, :])
        
        return mask_data
    
    def resize_mask(self, mask_data, image_size_tuple:tuple[int,int]= None):
        """
        Resize mask data using nearest-neighbor interpolation to preserve label integrity.
        Parameters:
        - mask_data: The mask data as a numpy array.
        - image_size_tuple: (H, W) in pixels to resize mask. 
        
        Returns:
        - A numpy array with dimensions (H, W).
        """
        def resize_mask_2D(mask_slice, image_size_tuple:tuple[int,int]):
            # Resize the mask slice
            resized_mask = transform.resize(
                mask_slice,
                image_size_tuple,
                order=0,  # nearest-neighbor interpolation to preserve label integrity
                preserve_range=True,
                mode='constant',
                anti_aliasing=False
            )
            return resized_mask
        
        def pad_to_square(mask_slice):
            height, width = mask_slice.shape
            if height > width:
                pad = (height - width) // 2
                padded_mask = np.pad(mask_slice, ((0, 0), (pad, height - width - pad)), mode='constant')
            elif width > height:
                pad = (width - height) // 2
                padded_mask = np.pad(mask_slice, ((pad, width - height - pad), (0, 0)), mode='constant')
            else:
                padded_mask = mask_slice  # already square
            return padded_mask
        
        if self.make_square:
            if len(mask_data.shape) == 2:
                mask_data = pad_to_square(mask_data)
            elif len(mask_data.shape) == 3:
                mask_data = np.array([pad_to_square(slice) for slice in mask_data])

        if image_size_tuple is None:
            image_size_tuple = self.image_size_tuple

        dims = len(np.shape(mask_data))
        if dims == 2:
            resized_masks = resize_mask_2D(mask_data, image_size_tuple)
        elif dims == 3:
            resized_masks = []
            for mask_slice in mask_data:
                resized_mask = resize_mask_2D(mask_slice, image_size_tuple)
                resized_masks.append(resized_mask)
            resized_masks = np.array(resized_masks)
        
        return resized_masks
    


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
    
    
    # -------------------- DATA TRANSFORMATION FUNCTIONS --------------------

    def prep_mask_step1(self, mask_data: np.ndarray) -> np.ndarray:   # prep_mask_study_specific_step1
        '''
        Apply preprocessing steps that are specific to the study to the mask data.
        Params:
        - crop_non_zero_slices_flag: True or False. Indicate whether to only keep slices with segmentations if mask is 3D.
        - mask_data: Supports 2D [H x W] and 3D [Slice x H x W] mask_data.
        '''
        # Remove unwanted labels as specified by 'remove_label_ids'
        mask_data = self.remove_labels(mask_data)
        # Optionally label areas of a specific feature (e.g., bone, cartilage, disc) as distinct instances
        mask_data, _ = self.label_instances(mask_data)
        # Remove small objects from the mask based on provided voxel and pixel thresholds
        mask_data = self.remove_small_objects(mask_data)
        # Convert data type to uint8
        mask_data = np.uint8(mask_data)
        
        z_indices = []
        if self.crop_non_zero_slices_flag and len(np.shape(mask_data))==3:
            # Identify slices in the mask that contain non-zero values to focus processing on relevant areas
            z_indices = find_non_zero_slices(mask_data)
            # Skip further processing for this file if no non-zero slices are found
            if len(z_indices) == 0:
                raise UserWarning("Mask does not have labels")
            # Crop the mask data to only include relevant non-zero slices
            mask_data = crop_data_to_non_zero_slices(mask_data, z_indices)

        return (mask_data, z_indices)

    def prep_mask_step2(self, mask_data: np.ndarray) -> np.ndarray:   # prep_mask_sam_specific_step2
        '''
        Apply preprocessing steps that are specific to sam to the mask data.
        Params:
        - mask_data: Supports 2D [H x W] and 3D [Slice x H x W] mask_data.
        '''
        # Resize the cropped mask data before saving
        preprocessed_mask = self.resize_mask(mask_data)
        # Convert data type to uint8
        preprocessed_mask = np.uint8(preprocessed_mask)
        return preprocessed_mask
    
    def preprocess_mask(self, mask_data: np.ndarray) -> np.ndarray:
        '''
        Apply all preprocessing steps to the mask data. 
        Params:
        - mask_data: Supports 2D [H x W] and 3D [Slice x H x W] mask_data.
        '''
        mask_data, z_indices = self.prep_mask_step1(mask_data)
        preprocessed_mask = self.prep_mask_step2(mask_data)
        return (preprocessed_mask, z_indices)
    

class ImagePrep():
    """
    Class of static variables and functions to prepare images for SAM/YOLO. Supports 2D or 3D mask input.
    
    Static Params:
    - image_size_tuple: (H, W) in pixels to resize images using cubic spline interpolation. This 
        will preserve image intensities.

    Input Params:
    - image_data: The image data as a numpy array. Supports 2D slices or 3D volumes.
    - z_indices: True or False. Indicate whether to crop volume to only slices specified. This makes it
        possible to only select slices with segmentations. (Only possible for 3D masks)
    """
    def __init__(self, image_size_tuple:tuple[int, int] = [], make_square:bool=False, ratio_resize:bool=False):
        self.image_size_tuple = image_size_tuple
        self.make_square = make_square
        self.ratio_resize = ratio_resize
    
    
    # -------------------- UTILITY FUNCTIONS --------------------

    def clip_image_data(self, image_data: np.ndarray, valid_pixels) -> np.ndarray:
        """Clip outliers beyond 0.5% and 99.5% intensity range."""
        lower_bound, upper_bound = np.percentile(valid_pixels, [0.5, 99.5]) #TODO - Why only consider valid pixels?
        if lower_bound == upper_bound:
            return np.full_like(image_data, fill_value=127, dtype=np.uint8)  # Handle uniform image data
        
        clipped_data = np.clip(image_data, lower_bound, upper_bound)
        return clipped_data

    def normalize_image_data(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image data so that intensity values range from 0 to 1."""
        normalized_data = (image_data - image_data.min()) / np.clip(image_data.max() - image_data.min(), a_min=1e-8, a_max=None)  # normalize to [0, 1]
        return normalized_data 
    
    def normalize_image_data_by_slice(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image data by each slice so that intensities range from 0 to 1."""

        dims = len(np.shape(image_data))
        if dims == 2:
            normed_images = self.normalize_image_data(image_data)
        elif dims == 3:
            normed_images = []
            for img_slice in image_data:
                normed_img = self.normalize_image_data(img_slice)
                normed_images.append(normed_img)
            normed_images = np.array(normed_images)
        return normed_images

    def resize_images(self, image_data: np.ndarray, image_size_tuple:tuple[int, int]=None, n_channels:int=0) -> np.ndarray:
        """Resize image data using cubic spline interpolation."""
        
        def resize_image_2D(img_slice, image_size_tuple:(int,int)):
            # Resize the normalized image slice
            resized_img = transform.resize(
                img_slice,
                image_size_tuple,
                order=3,  # cubic spline interpolation
                preserve_range=True,
                mode='constant',
                anti_aliasing=True
            )
            return resized_img
        
        def pad_to_square(img_slice):
            height, width = img_slice.shape[-2:]  # Only consider the last two dimensions
            if height > width:
                pad = (height - width) // 2
                padded_img = np.pad(img_slice, ((0, 0), (pad, height - width - pad)), mode='constant')
            elif width > height:
                pad = (width - height) // 2
                padded_img = np.pad(img_slice, ((pad, width - height - pad), (0, 0)), mode='constant')
            else:
                padded_img = img_slice  # already square
            return padded_img
        
        if image_size_tuple is None:
            image_size_tuple = self.image_size_tuple

        # original_dtype = image_data.dtype

        if self.ratio_resize:

        if self.make_square:
            if len(image_data.shape) == 2:
                image_data = pad_to_square(image_data)
            elif len(image_data.shape) == 3 and n_channels == 0:
                image_data = np.array([pad_to_square(slice) for slice in image_data])
            elif len(image_data.shape) == 3 and n_channels > 0:
                image_data = np.array([pad_to_square(slice) for slice in image_data])
            elif len(image_data.shape) == 4 and n_channels > 0:
                image_data = np.array([[pad_to_square(slice) for slice in img_channel] for img_channel in image_data])

        dims = len(np.shape(image_data))
        if (dims==2 and n_channels==0) or (dims==3 and n_channels>0):   
            resized_images = resize_image_2D(image_data, image_size_tuple)
        elif (dims==3 and n_channels==0) or (dims==4 and n_channels>0):
            resized_images = []
            for img_slice in image_data:
                resized_img = resize_image_2D(img_slice, image_size_tuple)
                resized_images.append(resized_img)
            resized_images = np.array(resized_images)
        

        # resized_images = resized_images.astype(original_dtype)

        return resized_images
    
    # -------------------- DATA TRANSFORMATION FUNCTIONS --------------------

    def prep_image_step1(self, image_data: np.ndarray, z_indices:list[int] = []) -> np.ndarray:  # prep_image_study_specific_step1
        '''
        Apply preprocessing steps that are specific to the study to the image data.
        Params:
        - image_data: Supports 2D [H x W] and 3D [Slice x H x W] image data.
        - z_indices: relevant slices to keep if image_data is 3D
        '''

        clipped_data = self.clip_image_data(image_data, image_data[image_data>0])
        normalized_data = np.uint8(self.normalize_image_data(clipped_data) *255)

        if any(z_indices):
            # Cropping the image data to match the cropped mask data
            normalized_data = crop_data_to_non_zero_slices(normalized_data, z_indices)
            # Ensure image has non-zero intensities
            if np.sum(normalized_data) == 0:
                raise ValueError('No non-zero intensities in image.')
        
        return normalized_data
    
    def prep_image_step2(self, normalized_data: np.ndarray) -> np.ndarray:  # prep_image_sam_specific_step2
        '''
        Apply preprocessing steps that are specific to sam to the image data.
        Params:
        - image_data: Supports 2D [H x W] and 3D [Slice x H x W] image data.
        '''
        # Resize and normalize the cropped image and mask data before saving
        resized_data = self.resize_images(normalized_data)

        # resized_data = np.uint8(resized_data)

        # Removed on 4/9/2024 - resized_normalized_data = self.normalize_image_data_by_slice(resized_data)
        resized_normalized_data = self.normalize_image_data_by_slice(resized_data)

        # import pdb; pdb.set_trace()

        # resized_normalized_data = np.uint8(resized_normalized_data)

        return resized_normalized_data
    
    def preprocess_image(self, image_data: np.ndarray, z_indices:list[int] = []) -> np.ndarray:
        '''
        Apply all preprocessing steps to the image data.
        Params:
        - image_data: Supports 2D [H x W] and 3D [Slice x H x W] image data.
        - z_indices: relevant slices to keep if image_data is 3D
        '''
        normalized_data = self.prep_image_step1(image_data, z_indices)

        print(f"normalized_data dtype: {normalized_data.dtype}")

        resized_normalized_data = self.prep_image_step2(normalized_data)

        # print(f"resized_normalized_data dtype: {resized_normalized_data.dtype}")

        return resized_normalized_data
    
# ---------------------- Bounding Box Creation ------------------------------------- #

def write_list_to_file(file_path, data_list):
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(str(item) + '\n')
            
def get_bounding_boxes(multiclass_mask, instance=False):

    # obtain unique labels in the image
    unique_labels = np.unique(multiclass_mask)
    bounding_boxes = []

    for label in unique_labels:
        # Skip background label
        if label == 0:
            continue  
        
        if instance:
            # Process each instance separately
            gt2D = np.uint8(multiclass_mask == label)  # Binary mask for chosen class
            labeled_array, num_features = scipy_label(gt2D)

            for component in range(1, num_features + 1):
                component_mask = labeled_array == component
                y_indices, x_indices = np.where(component_mask)

                # Compute the bounding box for the selected component
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)

                # Calculate center and dimensions
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min

                # Normalize
                height_sz, width_sz = multiclass_mask.shape
                bounding_boxes.append(
                    str(label) +
                    ' ' + str(center_x / width_sz) +
                    ' ' + str(center_y / height_sz) +
                    ' ' + str(width / width_sz) +
                    ' ' + str(height / height_sz) )
        else:
            # Original logic for bounding box of entire label
            indices = np.where(multiclass_mask == label)

            # Calculate bounding box coordinates
            min_row, min_col = np.min(indices[0]), np.min(indices[1])
            max_row, max_col = np.max(indices[0]), np.max(indices[1])

            # Calculate center and dimensions
            center_x = (min_col + max_col) / 2
            center_y = (min_row + max_row) / 2
            width = max_col - min_col
            height = max_row - min_row

            # Normalize
            height_sz, width_sz = multiclass_mask.shape
            bounding_boxes.append(
                str(label) +
                ' ' + str(center_x / width_sz) +
                ' ' + str(center_y / height_sz) +
                ' ' + str(width / width_sz) +
                ' ' + str(height / height_sz) )

    return bounding_boxes