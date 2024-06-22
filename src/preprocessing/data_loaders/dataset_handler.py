import numpy as np
import os
import re
#from skimage.measure import label as sk_label
import cc3d

import src.preprocessing.data_loaders.raw_file_loaders as FileLoader
from src.preprocessing.data_loaders.imorphic_loader import extract_imorphics_mask

class RawDataLoader():
    """
    Class that loads data from various datasets. Inherits the FileLoader class functions to load files with various extensions. "dataset_id" variable allows for additional dataset-specific loading capability.
    
    Args:
        dataset_id: String that contains a substring to identify dataset.
    """
    def __init__(self, data_transforms={}):
        self.transform_cfg = data_transforms

    def top_largest_cc(self, data, k_largest:int, connectivity:int):
        '''
        Label by connected components and keep k_largest largest non-background labels.
        '''
        # Get a labeling of the k largest objects in the image.
        # The output will be relabeled from 1 to N.
        largest_k_data, N = cc3d.largest_k(data, k=k_largest, connectivity=connectivity, delta=0, return_N=True)
        # Use original labels and update labels to zero if not part of k largest objects
        data = data * (largest_k_data > 0) 
        return data

    def transform_data(self, data):
        # Check whether to perform transforms
        if not self.transform_cfg:
            return data
        
        # Apply transforms sequentially
        for transform_name, transform_args in self.transform_cfg.items():
            #print(transform_name, transform_args)
            #import pdb; pdb.set_trace()
            if transform_name == 'transpose':
                data = np.transpose(data, transform_args) # args are the axes
            elif transform_name == 'swapaxes':
                data = np.swapaxes(data, transform_args[0], transform_args[1]) #args are the axes 
            elif transform_name == 'flip':
                data = np.flip(data, axis=transform_args) # args is the axis
            elif transform_name == 'rot90':
                data = np.rot90(data, axes=transform_args) #args are the axes 
            elif transform_name == 'pred_to_binary':
                data[data <= 0.5] = 0
                data[data > 0.5] = 1
            #elif transform_name == 'relabel': # not in use
            #    data[data == transform_args[0]] = transform_args[1]
            elif transform_name == 'remove_values_below':
                data[data<transform_args] = 0
            elif transform_name == 'drop_dim': # remove axes with dim 1
                if transform_args[0]==0:
                    data = np.squeeze(data[transform_args[1]])
                elif transform_args[0]==1:
                    data = np.squeeze(data[:,transform_args[1]])
                elif transform_args[0]==2:
                    data = np.squeeze(data[:,:,transform_args[1]])
                elif transform_args[0]==3:
                    data = np.squeeze(data[:,:,:,transform_args[1]])
                else:
                    raise ValueError('Invalid input for transform drop_dims')
            elif transform_name == 'top_cc_3D':
                data = self.top_largest_cc(data, transform_args, connectivity=26)
            elif transform_name == 'top_cc_2D':
                for slice_index in range(np.shape(data)[0]):
                    data[slice_index] = self.top_largest_cc(data[slice_index], transform_args, connectivity=8)
            # elif transform_name == 'combination_method':
            #     # The script will combine masks from one matrix
            #     # Note combining data from multiple files occurs elsewhere
            #     if (transform_args == 'combine_binary_label_dims') or (transform_args == 'add_multi_class_label_dims'):
            #         for dim in range(np.shape(data)[-1]):
            #             # Prepare data for combination
            #             if transform_args == 'combine_binary_label_dims':
            #                 label_counter = dim + 1  # Increment label for the next mask
            #                 labeled_data = np.where(data[...,dim] > 0, label_counter, 0)
            #             elif transform_args == 'add_multi_class_label_dims':
            #                 labeled_data = np.copy(data[...,dim])
            #             # Combine data with priority to first labels
            #             if dim == 0:
            #                 combined_data = labeled_data
            #             else:
            #                 # Conditional addition to avoid overwriting existing labels
            #                 combined_data = np.where(combined_data == 0, labeled_data, combined_data)
            #         data = np.copy(combined_data)
            elif transform_name == 'extract_imorphics_mask':
                data = extract_imorphics_mask(data)
            elif transform_name == 'dtype':
                n_vals = len(np.unique(data))
                n_type = data.dtype
                if transform_args == 'np.uint8':
                    data = data.astype(np.int8)
                elif transform_args == 'np.uint32':
                    data = data.astype(np.int32)
                if n_vals != len(np.unique(data)):
                    print(data.dtype, n_vals, len(np.unique(data)))
                    raise ValueError("Information lost during data conversion")
                #print(n_type, data.dtype, n_vals, len(np.unique(data)))
            elif transform_name == 'collapse_label_dim':
                data_ref = np.copy(data[:,:,:,transform_args[0]])
                label_counter_list = transform_args[1] 
                for index in range(np.shape(data_ref)[3]):
                    label_counter = label_counter_list[index]
                    if index == 0:
                        data = np.copy(data_ref[:,:,:,index]*label_counter)
                    else:
                        # Conditional addition to avoid overwriting existing labels
                        data = np.where(data == 0, data_ref[:,:,:,index]*label_counter, data)
        return data


    def get_npy(self, npy_path:str, key:str):
        """
        Load NumPy arrays and perform additional dataset-specific processing.
        """
        data = FileLoader.load_npy(npy_path, key)
        return data
    
    def get_npz(self, npz_path:str, key:str):
        """
        Load NumPy NPZ files and perform additional dataset-specific processing.
        """
        data = FileLoader.load_npz(npz_path, key)
        return data
    
    def get_h5(self, h5_path:str, key:str):
        """
        Load HDF5 files and perform additional dataset-specific processing.
        """
        data = FileLoader.load_h5(h5_path, key)
        return data
    
    def get_int2(self, img_path:str):
        """
        Load .int2 file and perform additional dataset-specific processing.
        """
        data = FileLoader.load_int2(img_path)
        return data
    
    def get_mat(self, mat_path:str, key:str):
        """
        Load .mat files and perform additional dataset-specific processing.
        """
        data = FileLoader.load_mat(mat_path, key)
        if 'oai' in mat_path.lower():
            data = FileLoader.load_mat(mat_path, key, struct_as_record=False)
        return data 

    def get_mhd(self, mhd_path:str):
        """
        Load .mhd files and perform additional dataset-specific processing.
        """
        # Load OAI-specific .mhd data - ZIB dataset
        data = FileLoader.load_mhd(mhd_path)
        return data
    
    def get_nifti(self, npy_path:str):
        """
        Load Nifti files and perform additional dataset-specific processing.
        """
        data = FileLoader.load_nifti(npy_path)
        return data

    def get_dcm(self, dcm_dir:str):
        """
        Load .dcm files and perform additional dataset-specific processing.
        """
        data = FileLoader.load_dcm(dcm_dir)
        return data
    
    def get_data_from_file(self, file_path:str, key:str='', transform_flag:bool=True):
        """
        Main function to load the data based on file extension and dataset.

        Params:
        - transform_flag: True/False. Either perform transform after loading file or directory.
        """
        # Get extension for files, and folders with dicoms
        if not os.path.isfile(file_path):
            return ValueError(f"File does not exist: {file_path}")
        
        # Extract the full extension for special cases (e.g., .h5.gz)
        _, full_extension = os.path.splitext(file_path.lower())

        # Handle compressed files (.gz) and other special cases
        if full_extension == '.gz':
            base_name, primary_extension = os.path.splitext(os.path.basename(file_path.lower())[:-3])
            extension = f"{primary_extension}.gz"
        else:
            base_name, extension = os.path.splitext(os.path.basename(file_path.lower()))

        if extension == '.npy':
            data = self.get_npy(file_path, key)
        elif extension in ['.npz', '.npz.gz']:
            data =  self.get_npz(file_path, key)
        elif extension in ['.h5', '.h5.gz','.hdf5']:
            data =  self.get_h5(file_path, key)
        elif extension == '.int2':
            data = self.get_int2(file_path)
        elif extension == '.mat':
            data =  self.get_mat(file_path, key)
        elif extension == '.mhd':
            data = self.get_mhd(file_path)
        elif extension in ['.nii', '.nii.gz']:
            data = self.get_nifti(file_path)
        # Add more conditions as needed

        if data is not None:
            if transform_flag == True:
                data = self.transform_data(data)
            return data
        return ValueError(f"Unsupported file format: {extension}")

    def get_data_from_dir(self, dir_path:str, key:str='', no_dicom_ext_flag=False):
        """
        Main function to combine slice files into a single 3D numpy array, handling various formats.
        """ 
        def alphanumeric_sort(strings):
            """
            Sort files by name without leading zeros. Supports 1, 2, ... 10, 11, instead of 1, 11, 2, 22, ...
            """
            def alphanumeric_key(string):
                return [int(s) if s.isdigit() else s for s in re.split('([0-9]+)', string)]
            return sorted(strings, key=alphanumeric_key)

        # Get extension of files in directory
        if not os.path.isdir(dir_path):
            return ValueError(f"Directory does not exist: {dir_path}")
        
        slice_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if not f.startswith('.')]
        extension = os.path.splitext(slice_files[0])[1].lower()
        # Check for any dicom in folder and disregard other files (i.e. .png, .mat)
        all_extensions = set([os.path.splitext(file)[1].lower() for file in slice_files])
        
        # Load dicoms
        if any(extension in ['.dcm', '.dcm.gz'] for extension in all_extensions) or no_dicom_ext_flag:
            data = self.get_dcm(dir_path)
            data = self.transform_data(data)
            return data
        # Load volume from slice files
        elif extension in ['.npy', '.h5', '.hdf5']:
            #slice_files are sorted using an alphanumeric_sort to support filenames without leading zeros.
            #sorted() is not used because it results in the incorrect order: 1, 11, 2, 22, ...
            slice_files = alphanumeric_sort(slice_files)
            slices = []
            for file_path in slice_files:
                slice_data = self.get_data_from_file(file_path, key, transform_flag=False)
                slices.append(slice_data)
            data = np.stack(slices, axis=-1)
            
            data = self.transform_data(data)
            return data 
        
        # Add more conditions as needed
        return ValueError(f"Unsupported file format: {extension}")

# ------------------------- MAIN FUNCTIONS -------------------------    
def get_data(data_path:str, key:str=None, data_transforms:str='', no_dicom_ext_flag:bool=False):
    """
    Loads data stored in various file formats from a single file or directory into 
    a np array, using the RawDataLoader class.
    """
    myRawDataLoader = RawDataLoader(data_transforms)

    # Determine if handling a directory, a single file, or a list of files
    # Process single file 
    if os.path.isfile(data_path):
        return myRawDataLoader.get_data_from_file(data_path, key)
    # Process a directory
    elif os.path.isdir(data_path):
        return myRawDataLoader.get_data_from_dir(data_path, key, no_dicom_ext_flag)
    # Process a list of files
    elif isinstance(data_path, list):
        raise TypeError("Use get_combined_data function to load multiple files.")
        
def get_combined_data(data_paths: list[str], key:str='', data_transforms:str=''):
    """
    Loads data from multiple file and combine them into a single volume.
    """
    def convert_dtype(data, data_type):
        n_vals = len(np.unique(data))
        if data_type == 'int8':
            data = data.astype(np.int8)
        elif data_type == 'int16':
            data = data.astype(np.int16)
        elif data_type == 'int64':
            data = data.astype(np.int64)
        # Check whether labels are retained
        if n_vals != len(np.unique(data)):
            print(data.dtype, n_vals, len(np.unique(data)))
            raise ValueError("Information lost during data conversion")
        return data

    # def DHAL_transforms(data):
    #     #To add humerous
    #     data = np.flip(data, axis=2)
    #     data = np.transpose(data,[2, 0, 1])
    #     data = data.astype(np.int8)
    #     return data

    combined_data = None
    label_counter = 1  # Start label counter for unique mask values
    
    for data_path in data_paths:
        data = get_data(data_path, key, data_transforms)

        # Prepare data for combination
        if data_transforms.get("combination_method", "") == "add_multi_class_labels":
            labeled_data = np.copy(data)
        else: # data_transforms.get("combination_method", "combine_binary_labels") == "":
            # Apply unique label to mask
            labeled_data = np.where(data > 0, label_counter, 0)
            label_counter += 1  # Increment label for the next mask

        # Combine data with priority to first labels
        if combined_data is None:
            combined_data = labeled_data
        else:
            # Conditional addition to avoid overwriting existing labels
            combined_data = np.where(combined_data == 0, labeled_data, combined_data)
    
    # Convert data type to match data variable
    combined_data = convert_dtype(combined_data, data.dtype)
    return combined_data

    
# ------------------------- FUNCTIONS TO VERIFY DATA -------------------------
def data_properties(data, content='mask'):
    """
    Extracts properties about the data into a dictionary.
    """
    assert content in ['image', 'mask'], ValueError('Invalid option for data content variable')
    
    def number_connected_components(data, connectivity=26):
        labeled_data, num_labels = cc3d.connected_components(data, connectivity=connectivity, return_N=True)
        return num_labels
    
    data_info = {
        'shape': np.shape(data),
        'max': np.max(data),
        'min': np.min(data),
        'number_unique_values': len(np.unique(data)),
        'type': data.dtype,
        'finite_values': np.all(np.isfinite(data)), #TODO # Replace NaNs and infs with zero or a specified value --> valid_vol =  np.nan_to_num(vol) 
        'masked_array': np.ma.is_masked(data), #TODO # Replace masked values with zero --> valid_vol = valid_vol.filled(0) 
    }

    if content == 'mask':
        data_info['unique_values'] = sorted(np.unique(data))
        #data_info['number_connected_components'] = number_connected_components(data) # longer runtime

    return data_info

def validate_data_properties(actual_properties:dict, expected_properties:dict):
    """
    Checks whether actual and expected data properties match.
    
    Params:
        actual_properties: Properties of data. Keys must match expected_properties.
        expected_properties: Properties we expect the data to have. Keys must match actual_properties.
    Returns:
        matching_properties(bool): True/False
    """
    matching_properties = True
    for property_name, expected_val in expected_properties.items():
        actual_val = actual_properties[property_name]
        #print(property_name, actual_val)
        if actual_val != expected_val:
            matching_properties = False
            #raise UserWarning(f'Unexpected data {property_name}\n\tExpected {expected_val}\n\tActual {actual_val}.')
            raise ValueError(f'Unexpected data {property_name}\n\tExpected {expected_val}\n\tActual {actual_val}.')
    return matching_properties

def validate_data(data, expected_properties:dict, content='mask'):
    """
    Determines whether data is consistent with what is expected.
    """
    actual_properties = data_properties(data, content)
    return validate_data_properties(actual_properties, expected_properties)