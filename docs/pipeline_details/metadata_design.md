# TBrecon - MICCAI Challenge Dataset Example

## Subject-level Dataset Metadata:

{'TBrecon-01-02-00047': {'subject_id': 'TBrecon-01-02-00047',
  'image_nifti': '/data/TBrecon3/Users/ghoyer/SAM_data/new_design_tests_TBrecon/tbrecon_nifti_img/TBrecon-01-02-00047.nii.gz',
  'mask_nifti': '/data/TBrecon3/Users/ghoyer/SAM_data/new_design_tests_TBrecon/tbrecon_nifti_mask/TBrecon-01-02-00047.nii.gz',
  'Sex': 'F',
  'Age': 42,
  'Weight': 70.31,
  'Dataset': 'TBrecon',
  'Anatomy': 'knee',
  'num_slices': 196,
  'mask_labels': {'0': 'background',
   '1': 'femoral cartilage',
   '2': 'tibial cartilage',
   '3': 'patellar cartilage',
   '4': 'femur',
   '5': 'tibia',
   '6': 'patella'},
  'field_strength': '3.0',
  'mri_sequence': '3D_CUBE',
  'Split': 'test'},

  'TBrecon-01-02-00007': {'subject_id': 'TBrecon-01-02-00007',
  'image_nifti': '/data/TBrecon3/Users/ghoyer/SAM_data/new_design_tests_TBrecon/tbrecon_nifti_img/TBrecon-01-02-00007.nii.gz',
  'mask_nifti': '/data/TBrecon3/Users/ghoyer/SAM_data/new_design_tests_TBrecon/tbrecon_nifti_mask/TBrecon-01-02-00007.nii.gz',
  'Sex': 'M',
  'Age': 24,
  'Weight': 81.0,
  'Dataset': 'TBrecon',
  'Anatomy': 'knee',
  'num_slices': 196,
  'mask_labels': {'0': 'background',
   '1': 'femoral cartilage',
   '2': 'tibial cartilage',
   '3': 'patellar cartilage',
   '4': 'femur',
   '5': 'tibia',
   '6': 'patella'},
  'field_strength': '3.0',
  'mri_sequence': '3D_CUBE',
  'Split': 'test'},
  ...
}

met_sub['TBrecon-01-02-00008'].keys() ->
dict_keys(['subject_id', 'image_nifti', 'mask_nifti', 'Sex', 'Age', 'Weight', 'Dataset', 'Anatomy', 'num_slices', 'mask_labels', 'field_strength', 'mri_sequence', 'Split'])



## Slice-level Dataset Metadata - Parquet file for each subject:

              subject_id slice_number  \
0    TBrecon-01-02-00008          000   
1    TBrecon-01-02-00008          001   
2    TBrecon-01-02-00008          002   
3    TBrecon-01-02-00008          003   
4    TBrecon-01-02-00008          004   
..                   ...          ...   
117  TBrecon-01-02-00008          117   
118  TBrecon-01-02-00008          118   
119  TBrecon-01-02-00008          119   
120  TBrecon-01-02-00008          120   
121  TBrecon-01-02-00008          121   

                                          npy_base_dir  \
0    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
1    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
2    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
3    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
4    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
..                                                 ...   
117  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
118  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
119  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
120  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
121  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   

                                        npy_image_path  \
0    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
1    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
2    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
3    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
4    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
..                                                 ...   
117  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
118  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
119  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
120  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   
121  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...   

                                         npy_mask_path  Dataset  mask_labels  
0    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  
1    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  
2    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  
3    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  
4    /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  
..                                                 ...      ...          ...  
117  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  
118  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  
119  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  
120  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  
121  /data/TBrecon3/Users/ghoyer/SAM_data/new_desig...  TBrecon  placeholder  

[122 rows x 7 columns]



## Stats metadata with slice-level details:
(Mock data)

ex: thigh_meta['OAI_902173454']['slices'].keys()

-> dict_keys(['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015'])

thigh_meta['OAI_902173454']['slices']['007']
->

{'AccessionNumber': '016610439902',
 'SOPInstanceUID': '1.2.82.9798798.342314.324.2343554.23434',
 'StudyInstanceUID': '1.3.114234.1234135.15315.1351235',
 'SeriesInstanceUID': '1.3.12.13251.12351325.123033',
 'series_desc': 'AX_T1_THIGH',
 'study_desc': 'OAI^MR^ENROLLMENT^THIGH',
 'TE': 10.0,
 'TR': 600.0,
 'flip_angle': 90.0,
 'ETL': 1,
 'field_strength': 2.89362,
 'scanner_name': '',
 'scanner_model': 'Trio',
 'slice_thickness': 5.0,
 'slice_spacing': 5.0,
 'pixel_spacing': [0.9765625, 0.9765625],
 'rows': 256,
 'columns': 512,
 'instanceNumber': 7,
 'slice_location': 45.0,
 'image_position_patient': [-233.53511, -125.0, 45.0]}


 ['slices']['001'] ->
 ...
  'slice_thickness': 5.0,
 'slice_spacing': 5.0,
 'pixel_spacing': [0.9765625, 0.9765625],
 'rows': 256,
 'columns': 512,
 'instanceNumber': 1,
 'slice_location': 75.0,
 'image_position_patient': [-233.53511, -125.0, 75.0]}


 ['slices']['015'] ->
  'slice_thickness': 5.0,
 'slice_spacing': 5.0,
 'pixel_spacing': [0.9765625, 0.9765625],
 'rows': 256,
 'columns': 512,
 'instanceNumber': 15,
 'slice_location': 5.0,
 'image_position_patient': [-233.53511, -125.0, 5.0]}