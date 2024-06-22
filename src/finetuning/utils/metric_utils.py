import json


def parse_image_name(img_name):
    """Parses the image name to extract subject ID and slice ID, increments slice ID by 1."""
    base_name = img_name.split('.')[0]  # Remove the file extension

    # Find the position of the last dash
    last_dash_index = base_name.rfind('-')

    # Split the base name at the last dash
    subj_id = base_name[:last_dash_index]
    slice_num = base_name[last_dash_index + 1:]  # Take the part after the last dash
   
    # Increment slice number by 1 and zero-pad to three digits
    incremented_slice_id = str(int(slice_num) + 1).zfill(3)
    return subj_id, incremented_slice_id

def load_meta(metadata_path, representative_slice=True):
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    
    if representative_slice:
        # Initialize reduced metadata dictionary
        reduced_metadata = {}
        
        for subj_id, subject_data in metadata.items():
            slices = subject_data['slices']
            
            # Ensure slices is a dictionary
            if slices and isinstance(slices, dict):
                # Get the first slice's key (slice name)
                first_slice_key = next(iter(slices))
                representative_slice = slices[first_slice_key]
                
                # Extract required fields from the representative slice
                slice_thickness = representative_slice.get('slice_thickness')
                slice_spacing = representative_slice.get('slice_spacing')
                pixel_spacing = representative_slice.get('pixel_spacing')
                rows = representative_slice.get('rows')
                columns = representative_slice.get('columns')
                
                # Add extracted fields to the reduced metadata for the subject
                reduced_metadata[subj_id] = {
                    'slice_thickness': slice_thickness,
                    'slice_spacing': slice_spacing,
                    'pixel_spacing': pixel_spacing,
                    'rows': rows,
                    'columns': columns
                }

        return reduced_metadata
    else:
        # Reduce the loaded metadata to only what is necessary to minimize memory usage
        reduced_metadata = {subj_id: subject_data['slices'] for subj_id, subject_data in metadata.items()}
        return reduced_metadata

def extract_meta(metadata_dict, subj_id, fields, slice_id=None):
    """
    Extracts metadata for a specific subject and returns the requested fields.
    
    Args:
        metadata_dict (dict): The metadata dictionary.
        subj_id (str): The subject ID.
        fields (list): List of fields to extract.
        slice_id (str, optional): The slice name ID. Required if representative_slice is False.
        
    Returns:
        dict: A dictionary of the requested fields and their values.
    """
    if slice_id is None:
        subject_meta = metadata_dict.get(subj_id, {})
        extracted_meta = {field: subject_meta.get(field) for field in fields}
        return extracted_meta
    else:
        subject_meta = metadata_dict.get(subj_id, {}).get(slice_id, {})
        extracted_meta = {field: subject_meta.get(field) for field in fields}
        return extracted_meta



# ---------------------------------------

# def load_meta(metadata_path):
#     with open(metadata_path, 'r') as file:
#         metadata = json.load(file)
    
#     # Initialize reduced metadata dictionary
#     reduced_metadata = {}
    
#     for subj_id, subject_data in metadata.items():
#         slices = subject_data['slices']
        
#         # Ensure slices is a dictionary
#         if slices and isinstance(slices, dict):
#             # Get the first slice's key (slice name)
#             first_slice_key = next(iter(slices))
#             representative_slice = slices[first_slice_key]
            
#             # Extract required fields from the representative slice
#             slice_thickness = representative_slice.get('slice_thickness')
#             slice_spacing = representative_slice.get('slice_spacing')
#             pixel_spacing = representative_slice.get('pixel_spacing')
#             rows = representative_slice.get('rows')
#             columns = representative_slice.get('columns')
            
#             # Add extracted fields to the reduced metadata for the subject
#             reduced_metadata[subj_id] = {
#                 'slice_thickness': slice_thickness,
#                 'slice_spacing': slice_spacing,
#                 'pixel_spacing': pixel_spacing,
#                 'rows': rows,
#                 'columns': columns
#             }

#     return reduced_metadata

# -----------------------------------------
# def extract_meta(metadata_dict, subj_id, fields):
#     """
#     Extracts metadata for a specific subject and returns the requested fields.
    
#     Args:
#         metadata_dict (dict): The metadata dictionary.
#         subj_id (str): The subject ID.
#         fields (list): List of fields to extract.
        
#     Returns:
#         dict: A dictionary of the requested fields and their values.
#     """
#     subject_meta = metadata_dict.get(subj_id, {})
#     extracted_meta = {field: subject_meta.get(field) for field in fields}
#     return extracted_meta
