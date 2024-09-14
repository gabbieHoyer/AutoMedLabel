Overview of different dataset metadata formats:
------------------------------------------------
------------------------------------------------

TBrecon Knee
--------

ML metadata
-----------
{
    "TBrecon-01-02-00007": {
        "subject_id": "TBrecon-01-02-00007",
        "image_nifti": "/data/mskdata/standardized/TBrecon/knee/3D_CUBE/nifti/imgs/TBrecon-01-02-00007.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/TBrecon/knee/3D_CUBE/nifti/masks/TBrecon-01-02-00007.nii.gz",
        "Sex": "M",
        "Age": 24,
        "Weight": 81.0,
        "Dataset": "TBrecon",
        "Anatomy": "knee",
        "num_slices": 196,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "3D_CUBE",
        "Split": "train"
    },

Stats metadata
--------------
{
    "TBrecon-01-02-00007": {
        "Dataset": "TBrecon",
        "Anatomy": "knee",
        "slices": {
            "IM00001": {
                "PID": "48d79531-9be3-4e28-b0f4-37251f420c76",
                "SOPInstanceUID": "2.25.151277390378651331885135333637139971849",
                "StudyInstanceUID": "2.25.87906126917510228044835687260582844048",
                "SeriesInstanceUID": "2.25.47404270657918172659646576229329655646",
                "series_desc": "TBrecon-01-02-00007",
                "slice_thickness": 0.6,
                "pixel_spacing": [
                    0.293,
                    0.293
                ],
                "rows": 512,
                "columns": 512,
                "instanceNumber": 1
            },

Metadata Summary csv
--------------------
,PID,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,slice_thickness,pixel_spacing,rows,columns,instanceNumber

TBrecon-01-02-00007,48d79531-9be3-4e28-b0f4-37251f420c76,2.25.45115352966195051293200865569158462772,2.25.87906126917510228044835687260582844048,2.25.47404270657918172659646576229329655646,TBrecon-01-02-00007,0.6,"['0.293', '0.293']",512,512,196

*************************************************************

OAI Knee
--------

ML metadata
-----------
{
    "OAI_9040390": {
        "subject_id": "OAI_9040390",
        "image_nifti": "/data/mskdata/standardized/OAI/knee/3D_DESS/Imorphics/nifti/imgs/OAI_9040390.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/OAI/knee/3D_DESS/Imorphics/nifti/masks/OAI_9040390.nii.gz",
        "Sex": "M",
        "Age": 48,
        "Weight": 81.7,
        "Dataset": "OAI_imorphics",
        "Anatomy": "knee",
        "num_slices": 160,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "3D_DESS",
        "Split": "train"
    },

Stats metadata
--------------
{
    "OAI_9040390": {
        "Dataset": "OAI_imorphics",
        "Anatomy": "knee",
        "slices": {
            "001": {
                "AccessionNumber": "016610142509",
                "SOPInstanceUID": "1.2.826.0.1.3680043.2.429.0.166101425.1100906065.1.3.1",
                "StudyInstanceUID": "1.3.12.2.1107.5.2.13.20576.4.0.10563235014321273",
                "SeriesInstanceUID": "1.3.12.2.1107.5.2.13.20576.4.0.10575855114886920",
                "series_desc": "SAG_3D_DESS_RIGHT",
                "study_desc": "OAI^MR^ENROLLMENT^RIGHT",
                "TE": 4.71,
                "TR": 16.32,
                "flip_angle": 25.0,
                "ETL": 1,
                "field_strength": 2.89362,
                "scanner_model": "Trio",
                "slice_thickness": 0.69999999,
                "pixel_spacing": [
                    0.36458333,
                    0.36458333
                ],
                "rows": 384,
                "columns": 384
            },

Metadata Summary csv
--------------------
,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,acquisition_type,TE,TR,flip_angle,ETL,field_strength,scanner_model,slice_thickness,pixel_spacing,rows,columns,imaging_frequency,pixel_bandwidth,coil_array,acquisition_matrix,SAR,percent_phase_FOV,phase_encoding_steps,percent_sampling
OAI_9040390,016610142509,1.2.826.0.1.3680043.2.429.0.166101425.1100906065.1.3.160,1.3.12.2.1107.5.2.13.20576.4.0.10563235014321273,1.3.12.2.1107.5.2.13.20576.4.0.10575855114886920,SAG_3D_DESS_RIGHT,OAI^MR^ENROLLMENT^RIGHT,3D,4.71,16.32,25.0,1,2.89362,Trio,0.69999999,"['0.36458333', '0.36458333']",384,384,123.22468,185.0,Extremity_2,"[0, 384, 307, 0]",0.0011984905,100.0,269,79.947917

*************************************************************

P50 MAPSS
---------

ML metadata
-----------
{
    "ACL004": {
        "subject_id": "ACL004",
        "image_nifti": "/data/mskdata/standardized/P50/knee/MAPSS_echo1/cartilage_compartments/nifti/imgs/ACL004.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/P50/knee/MAPSS_echo1/cartilage_compartments/nifti/masks/ACL004.nii.gz",
        "Sex": "M",
        "Age": "029Y",
        "Weight": 84.822,
        "Dataset": "P50_MAPSS",
        "Anatomy": "knee",
        "num_slices": 24,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "MAPSS-echo1",
        "Split": "train",
        "T1rho_map_nifti": "/data/mskdata/standardized/P50/knee/MAPSS_echo1/cartilage_compartments/nifti/T1rho_maps/ACL004.nii.gz",
        "T2_map_nifti": "/data/mskdata/standardized/P50/knee/MAPSS_echo1/cartilage_compartments/nifti/T2_maps/ACL004.nii.gz"
    },

Stats metadata
--------------
{
    "ACL004": {
        "Dataset": "P50_MAPSS",
        "Anatomy": "knee",
        "slices": {
            "E4363S5I1": {
                "PID": "51363949",
                "AccessionNumber": "7941832",
                "SOPInstanceUID": "1.2.840.113619.2.244.6945.1127392.28858.1323293250.560",
                "StudyInstanceUID": "1.2.840.113745.101000.1150000.40873.6703.13621794",
                "SeriesInstanceUID": "1.2.840.113619.2.244.6945.1127392.31494.1323292430.394",
                "series_desc": "MAPS T1rho T2 ARC- Right",
                "study_desc": "MR LOWER EXTREMITY JOINT,RIGHT",
                "TE": 0.0,
                "TR": 9.014,
                "flip_angle": 60.0,
                "ETL": 1,
                "field_strength": 3.0,
                "receive_coil": "HD TR Knee PA",
                "scanner_name": "CB-3TMR",
                "scanner_model": "Signa HDxt",
                "slice_thickness": 4.0,
                "slice_spacing": 4.0,
                "pixel_spacing": [
                    0.5469,
                    0.5469
                ],
                "rows": 256,
                "columns": 256,
                "instanceNumber": 1
            },

Metadata Summary csv
--------------------
,PID,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,TE,TR,flip_angle,ETL,field_strength,receive_coil,scanner_name,scanner_model,slice_thickness,slice_spacing,pixel_spacing,rows,columns,instanceNumber,acquisition_type,imaging_frequency,pixel_bandwidth,acquisition_matrix,percent_phase_FOV,percent_sampling
ACL004,51363949,7941832,1.2.840.113619.2.244.6945.1127392.28858.1323293250.560,1.2.840.113745.101000.1150000.40873.6703.13621794,1.2.840.113619.2.244.6945.1127392.31494.1323292430.394,MAPS T1rho T2 ARC- Right,"MR LOWER EXTREMITY JOINT,RIGHT",0.0,9.014,60.0,1,3.0,HD TR Knee PA,CB-3TMR,Signa HDxt,4.0,4.0,"['0.5469', '0.5469']",256,256,1,3D,127.717771,488.281,"[0, 256, 128, 0]",100.0,68.471

*************************************************************


AFACL MAPSS
-----------

ML metadata
-----------

    "AFACL1003-01": {
        "subject_id": "AFACL1003-01",
        "image_nifti": "/data/mskdata/standardized/AFACL/knee/MAPSS_echo1/cartilage_compartments/nifti/imgs/AFACL1003-01.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/AFACL/knee/MAPSS_echo1/cartilage_compartments/nifti/masks/AFACL1003-01.nii.gz",
        "Sex": "F",
        "Age": "054Y",
        "Weight": 86.18,
        "Dataset": "P50_MAPSS",
        "Anatomy": "knee",
        "num_slices": 24,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "MAPSS-echo1",
        "Split": "test",
        "T1rho_map_nifti": "/data/mskdata/standardized/AFACL/knee/MAPSS_echo1/cartilage_compartments/nifti/T1rho_maps/AFACL1003-01.nii.gz",
        "T2_map_nifti": "/data/mskdata/standardized/AFACL/knee/MAPSS_echo1/cartilage_compartments/nifti/T2_maps/AFACL1003-01.nii.gz"
    },

Stats metadata
--------------
{
    "AFACL1003-01": {
        "Dataset": "P50_MAPSS",
        "Anatomy": "knee",
        "slices": {
            "IM_S1_20140128_I1": {
                "PID": "AFACL1003-01",
                "AccessionNumber": "6559990443058990",
                "SOPInstanceUID": "9999.98881456193322655297920521056239977398",
                "StudyInstanceUID": "9999.213775225419700990720932675219793993944",
                "SeriesInstanceUID": "9999.157468880174111627406787663415798864967",
                "series_desc": "Sagittal 3D T1p and T2 MAPSS Right",
                "study_desc": "e+1 MR LOWER EXTREMITY W/O CON,BIL",
                "TE": 0.0,
                "TR": 5.376,
                "flip_angle": 60.0,
                "ETL": 1,
                "field_strength": 3.0,
                "receive_coil": "HD TR Knee PA",
                "scanner_name": "OIMR1",
                "slice_thickness": 4.0,
                "slice_spacing": 4.0,
                "pixel_spacing": [
                    0.5469,
                    0.5469
                ],
                "rows": 256,
                "columns": 256,
                "instanceNumber": 1
            },

Metadata Summary csv
--------------------
,PID,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,TE,TR,flip_angle,ETL,field_strength,receive_coil,scanner_name,slice_thickness,slice_spacing,pixel_spacing,rows,columns,instanceNumber,acquisition_type,imaging_frequency,pixel_bandwidth,acquisition_matrix,percent_phase_FOV,percent_sampling
AFACL1003-01,AFACL1003-01,6559990443058990,9999.71087959948234267185006956896385194967,9999.213775225419700990720932675219793993944,9999.157468880174111627406787663415798864967,Sagittal 3D T1p and T2 MAPSS Right,"e+1 MR LOWER EXTREMITY W/O CON,BIL",0.0,5.376,60.0,1,3.0,HD TR Knee PA,OIMR1,4.0,4.0,"['0.5469', '0.5469']",256,256,99,3D,127.763372,488.281,"[0, 256, 128, 0]",100.0,68.471

*************************************************************


DHAL Shoulder
-------------

ML metadata
-----------

    "E23609": {
        "subject_id": "E23609",
        "image_nifti": "/data/mskdata/standardized/DHAL/shoulder/T2/nifti/imgs/E23609.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/DHAL/shoulder/T2/nifti/masks/E23609.nii.gz",
        "Sex": "M",
        "Age": "075Y",
        "Weight": 90.72,
        "Dataset": "DHAL",
        "Anatomy": "shoulder",
        "num_slices": 248,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "3D_CUBE",
        "Split": "train"
    },

Stats metadata
--------------
{
    "E23605": {
        "Dataset": "DHAL",
        "Anatomy": "shoulder",
        "slices": {
            "E23605S4I149": {
                "AccessionNumber": "12488459",
                "SOPInstanceUID": "1.2.840.113619.2.408.14196467.2651602.31151.1537193338.268",
                "StudyInstanceUID": "1.2.124.113532.80.22017.45499.20180913.160212.120604218",
                "SeriesInstanceUID": "1.2.840.113619.2.408.14196467.2651602.660.1537193322.698",
                "series_desc": "Roland CUBE (1MM)",
                "study_desc": "MR SHOULDER WITHOUT CONTRAST, LEFT",
                "TE": 49.565,
                "TR": 1352.0,
                "flip_angle": 90.0,
                "ETL": 30,
                "field_strength": 3.0,
                "receive_coil": "GEM Flex Medium",
                "scanner_name": "OIMR1",
                "scanner_model": "DISCOVERY MR750",
                "slice_thickness": 1.0,
                "slice_spacing": 0.499999,
                "pixel_spacing": [
                    0.375,
                    0.375
                ],
                "rows": 512,
                "columns": 512
            },

Metadata Summary csv
--------------------
,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,TE,TR,flip_angle,ETL,field_strength,receive_coil,scanner_name,scanner_model,slice_thickness,slice_spacing,pixel_spacing,rows,columns,acquisition_type,imaging_frequency,pixel_bandwidth,acquisition_matrix,SAR,percent_phase_FOV,percent_sampling
E23605,12488459,1.2.840.113619.2.408.14196467.2651602.31151.1537193338.268,1.2.124.113532.80.22017.45499.20180913.160212.120604218,1.2.840.113619.2.408.14196467.2651602.660.1537193322.698,Roland CUBE (1MM),"MR SHOULDER WITHOUT CONTRAST, LEFT",49.565,1352.0,90.0,30,3.0,GEM Flex Medium,OIMR1,DISCOVERY MR750,1.0,0.499999,"['0.375', '0.375']",512,512,3D,127.763436,195.312,"[192, 0, 0, 192]",1.4932,90.0,100.0

*************************************************************


KICK Hip
---------

ML metadata
-----------
{
    "K001": {
        "subject_id": "K001",
        "image_nifti": "/data/mskdata/standardized/KICK/hip/3D_CUBE/nifti/imgs/K001.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/KICK/hip/3D_CUBE/nifti/masks/K001.nii.gz",
        "Sex": "M",
        "Age": "061Y",
        "Weight": 63.5,
        "Dataset": "KICK",
        "Anatomy": "hip",
        "num_slices": 339,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "3D_CUBE",
        "Split": "train"
    },

Stats metadata
--------------
{
    "K001": {
        "Dataset": "KICK",
        "Anatomy": "hip",
        "slices": {
            "CUBE_harmonized_512_0001": {
                "AccessionNumber": "10021651282",
                "SOPInstanceUID": "1.3.6.1.4.1.9590.100.1.2.368677878542462647222698490590693580365",
                "StudyInstanceUID": "1.2.124.113532.80.22185.43466.20211201.132552.2653318",
                "SeriesInstanceUID": "1.3.6.1.4.1.19291.2.1.2.220169414560248671665898515",
                "series_desc": "Sag RT HIP",
                "study_desc": "MR HIP WITHOUT CONTRAST, RIGHT",
                "TE": 20.702,
                "TR": 1202.0,
                "flip_angle": 90.0,
                "ETL": 30,
                "field_strength": 3.0,
                "receive_coil": "30AA+60PA",
                "scanner_name": "UCSFCBMR7",
                "scanner_model": "Horos",
                "slice_thickness": 0.3125,
                "slice_spacing": 0.4,
                "pixel_spacing": [
                    0.3125,
                    0.3125
                ],
                "rows": 512,
                "columns": 512
            },

Metadata Summary csv
--------------------
,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,TE,TR,flip_angle,ETL,field_strength,receive_coil,scanner_name,scanner_model,slice_thickness,slice_spacing,pixel_spacing,rows,columns,acquisition_type,imaging_frequency,pixel_bandwidth,acquisition_matrix,SAR,percent_phase_FOV,percent_sampling
K001,10021651282,1.3.6.1.4.1.9590.100.1.2.368677878542462647222698490590693580365,1.2.124.113532.80.22185.43466.20211201.132552.2653318,1.3.6.1.4.1.19291.2.1.2.220169414560248671665898515,Sag RT HIP,"MR HIP WITHOUT CONTRAST, RIGHT",20.702,1202.0,90.0,30,3.0,30AA+60PA,UCSFCBMR7,Horos,0.3125,0.4,"['0.3125', '0.3125']",512,512,3D,127.769173,122.07,"[0, 200, 200, 0]",1.4948,200.0,100.0

*************************************************************


Spine UH2 T1ax
---------------

ML metadata
-----------
{
    "BACPAC_MDAI_t1ax_001": {
        "subject_id": "BACPAC_MDAI_t1ax_001",
        "image_nifti": "/data/mskdata/standardized/BACPAC_UH2/lumbar_spine/T1_ax/nifti/imgs/BACPAC_MDAI_t1ax_001.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/BACPAC_UH2/lumbar_spine/T1_ax/nifti/masks/BACPAC_MDAI_t1ax_001.nii.gz",
        "Sex": "M",
        "Age": "062Y",
        "Weight": 86.18,
        "Dataset": "BACPAC_UH2_T1ax",
        "Anatomy": "lumbar_spine",
        "num_slices": 66,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "t1_ax",
        "Split": "train"
    },

Stats metadata
--------------
{
    "BACPAC_MDAI_t1ax_001": {
        "Dataset": "BACPAC_UH2_T1ax",
        "Anatomy": "lumbar_spine",
        "slices": {
            "1.3.6.1.4.1.20319.190895674932276385098066107081273990639": {
                "PID": "iNpePwm6Ri",
                "AccessionNumber": "xCIehvlJFZ",
                "SOPInstanceUID": "1.3.6.1.4.1.20319.190895674932276385098066107081273990639",
                "StudyInstanceUID": "1.3.6.1.4.1.20319.145156039351403154368892453748937019687",
                "SeriesInstanceUID": "1.3.6.1.4.1.20319.75905961172453233690096690131566835022",
                "series_desc": "AX T1 FSE",
                "study_desc": "MR LUMBAR SPINE WITHOUT CONTRAST",
                "TE": 12.66,
                "TR": 762.0,
                "flip_angle": 120.0,
                "ETL": 4,
                "field_strength": 3.0,
                "receive_coil": "HNS CTL456",
                "scanner_name": "OIMR1",
                "scanner_model": "DISCOVERY MR750",
                "slice_thickness": 3.0,
                "slice_spacing": 3.0,
                "pixel_spacing": [
                    0.3516,
                    0.3516
                ],
                "rows": 512,
                "columns": 512,
                "instanceNumber": 61
            },

Metadata Summary csv
--------------------
,PID,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,TE,TR,flip_angle,ETL,field_strength,receive_coil,scanner_name,scanner_model,slice_thickness,slice_spacing,pixel_spacing,rows,columns,instanceNumber,acquisition_type,imaging_frequency,pixel_bandwidth,acquisition_matrix,SAR,percent_phase_FOV,percent_sampling
BACPAC_MDAI_t1ax_001,iNpePwm6Ri,xCIehvlJFZ,1.3.6.1.4.1.20319.190895674932276385098066107081273990639,1.3.6.1.4.1.20319.145156039351403154368892453748937019687,1.3.6.1.4.1.20319.75905961172453233690096690131566835022,AX T1 FSE,MR LUMBAR SPINE WITHOUT CONTRAST,12.66,762.0,120.0,4,3.0,HNS CTL456,OIMR1,DISCOVERY MR750,3.0,3.0,"['0.3516', '0.3516']",512,512,61,2D,127.763262,162.773,"[256, 0, 0, 256]",3.4998,100.0,100.0

*************************************************************

Spine UH2 T2ax
---------------

ML metadata
-----------
{
    "BACPAC_MDAI_t2ax_001": {
        "subject_id": "BACPAC_MDAI_t2ax_001",
        "image_nifti": "/data/mskdata/standardized/BACPAC_UH2/lumbar_spine/T2_ax/nifti/imgs/BACPAC_MDAI_t2ax_001.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/BACPAC_UH2/lumbar_spine/T2_ax/nifti/masks/BACPAC_MDAI_t2ax_001.nii.gz",
        "Sex": "M",
        "Age": "066Y",
        "Weight": 67.0,
        "Dataset": "BACPAC_UH2_T2ax",
        "Anatomy": "lumbar_spine",
        "num_slices": 41,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "t2_ax",
        "Split": "train"
    },

Stats metadata
--------------
{
    "BACPAC_MDAI_t2ax_001": {
        "Dataset": "BACPAC_UH2_T2ax",
        "Anatomy": "lumbar_spine",
        "slices": {
            "1.3.6.1.4.1.20319.3892415514154423259428759231148315822": {
                "PID": "fLO3wkP0Sr",
                "AccessionNumber": "bhIXh733Fb",
                "SOPInstanceUID": "1.3.6.1.4.1.20319.3892415514154423259428759231148315822",
                "StudyInstanceUID": "1.3.6.1.4.1.20319.2653187360960748096443408769311094280",
                "SeriesInstanceUID": "1.3.6.1.4.1.20319.237720352171536956327946131586159486656",
                "series_desc": "T2W_AX lsp",
                "study_desc": "MR THORACIC SPINE, LUMBAR SPINE WITHOUT CONTRAST",
                "TE": 120.302,
                "TR": 3485.32250976562,
                "flip_angle": 90.0,
                "ETL": 42,
                "field_strength": 1.5,
                "receive_coil": "SENSE-Spine",
                "scanner_name": "UCSF-MR4",
                "scanner_model": "Achieva",
                "slice_thickness": 4.0,
                "slice_spacing": 5.0,
                "pixel_spacing": [
                    0.90277779102325,
                    0.90277779102325
                ],
                "rows": 288,
                "columns": 288,
                "instanceNumber": 35
            },

Metadata Summary csv
--------------------
,PID,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,TE,TR,flip_angle,ETL,field_strength,receive_coil,scanner_name,scanner_model,slice_thickness,slice_spacing,pixel_spacing,rows,columns,instanceNumber,acquisition_type,imaging_frequency,pixel_bandwidth,acquisition_matrix,SAR,percent_phase_FOV,percent_sampling
BACPAC_MDAI_t2ax_001,fLO3wkP0Sr,bhIXh733Fb,1.3.6.1.4.1.20319.3892415514154423259428759231148315822,1.3.6.1.4.1.20319.2653187360960748096443408769311094280,1.3.6.1.4.1.20319.237720352171536956327946131586159486656,T2W_AX lsp,"MR THORACIC SPINE, LUMBAR SPINE WITHOUT CONTRAST",120.302,3485.32250976562,90.0,42,1.5,SENSE-Spine,UCSF-MR4,Achieva,4.0,5.0,"['0.90277779102325', '0.90277779102325']",288,288,35,2D,63.897405,405.0,"[0, 260, 224, 0]",1.99969732761383,100.0,86.2820510864257

*************************************************************

Spine UH2 T1sag vert and disc
-----------------------------

ML metadata
-----------
{
    "BACPAC_MDAI_t1sag_001": {
        "subject_id": "BACPAC_MDAI_t1sag_001",
        "image_nifti": "/data/mskdata/standardized/BACPAC_UH2/lumbar_spine/T1_sag/vert_and_disc/nifti/imgs/BACPAC_MDAI_t1sag_001.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/BACPAC_UH2/lumbar_spine/T1_sag/vert_and_disc/nifti/masks/BACPAC_MDAI_t1sag_001.nii.gz",
        "Sex": "F",
        "Age": "055Y",
        "Weight": 90.72,
        "Dataset": "BACPAC_UH2_T1sag",
        "Anatomy": "lumbar_spine",
        "num_slices": 22,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "t1_sag",
        "Split": "train"
    },

Stats metadata
--------------
{
    "BACPAC_MDAI_t1sag_001": {
        "Dataset": "BACPAC_UH2_T1sag",
        "Anatomy": "lumbar_spine",
        "slices": {
            "1.3.6.1.4.1.20319.105231397917641909599069292771835977198": {
                "PID": "bv5iTBRtZF",
                "AccessionNumber": "LrwkNRFqWG",
                "SOPInstanceUID": "1.3.6.1.4.1.20319.105231397917641909599069292771835977198",
                "StudyInstanceUID": "1.3.6.1.4.1.20319.101174582855801956297716057908798760821",
                "SeriesInstanceUID": "1.3.6.1.4.1.20319.118722658481494274766396636560923165460",
                "series_desc": "SAG T1 Lspine",
                "study_desc": "MR LUMBAR SPINE WITHOUT CONTRAST",
                "TE": 9.832,
                "TR": 576.0,
                "flip_angle": 111.0,
                "ETL": 4,
                "field_strength": 3.0,
                "receive_coil": "HDCTL 456",
                "scanner_name": "mrc2",
                "scanner_model": "Signa HDxt",
                "slice_thickness": 3.0,
                "slice_spacing": 3.0,
                "pixel_spacing": [
                    0.4883,
                    0.4883
                ],
                "rows": 512,
                "columns": 512
            },

Metadata Summary csv
--------------------
,PID,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,TE,TR,flip_angle,ETL,field_strength,receive_coil,scanner_name,scanner_model,slice_thickness,slice_spacing,pixel_spacing,rows,columns,acquisition_type,imaging_frequency,pixel_bandwidth,acquisition_matrix,SAR,percent_phase_FOV,percent_sampling
BACPAC_MDAI_t1sag_001,bv5iTBRtZF,LrwkNRFqWG,1.3.6.1.4.1.20319.105231397917641909599069292771835977198,1.3.6.1.4.1.20319.101174582855801956297716057908798760821,1.3.6.1.4.1.20319.118722658481494274766396636560923165460,SAG T1 Lspine,MR LUMBAR SPINE WITHOUT CONTRAST,9.832,576.0,111.0,4,3.0,HDCTL 456,mrc2,Signa HDxt,3.0,3.0,"['0.4883', '0.4883']",512,512,2D,127.713839,139.492,"[320, 0, 0, 224]",2.9989,100.0,100.0

*************************************************************

Spine UH2 T1sag vert only
-------------------------

ML metadata
-----------
{
    "BACPAC_MDAI_t1sag_032": {
        "subject_id": "BACPAC_MDAI_t1sag_032",
        "image_nifti": "/data/mskdata/standardized/BACPAC_UH2/lumbar_spine/T1_sag/vert_only/nifti/imgs/BACPAC_MDAI_t1sag_032.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/BACPAC_UH2/lumbar_spine/T1_sag/vert_only/nifti/masks/BACPAC_MDAI_t1sag_032.nii.gz",
        "Sex": "F",
        "Age": "037Y",
        "Weight": 70.31,
        "Dataset": "BACPAC_UH2_T1sag_vert",
        "Anatomy": "lumbar_spine",
        "num_slices": 20,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "t1_sag",
        "Split": "train"
    },

Stats metadata
--------------
{
    "BACPAC_MDAI_t1sag_023": {
        "Dataset": "BACPAC_UH2_T1sag_vert",
        "Anatomy": "lumbar_spine",
        "slices": {
            "1.3.6.1.4.1.20319.227560727460099300023534241017731357428": {
                "PID": "YkHJhdPAoL",
                "AccessionNumber": "75J7ITsLkW",
                "SOPInstanceUID": "1.3.6.1.4.1.20319.227560727460099300023534241017731357428",
                "StudyInstanceUID": "1.3.6.1.4.1.20319.221087568742653321421428832991626143074",
                "SeriesInstanceUID": "1.3.6.1.4.1.20319.72915936643522625566197560692047820894",
                "series_desc": "t1_tse_sag",
                "study_desc": "MR LUMBAR SPINE WITH AND WITHOUT CONTRAST",
                "TE": 9.3,
                "TR": 518.0,
                "flip_angle": 140.0,
                "ETL": 3,
                "field_strength": 3.0,
                "scanner_name": "UCSF-ZMOB3T",
                "scanner_model": "Verio",
                "slice_thickness": 3.0,
                "slice_spacing": 3.0,
                "pixel_spacing": [
                    0.75,
                    0.75
                ],
                "rows": 320,
                "columns": 320
            },

Metadata Summary csv
--------------------
,PID,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,TE,TR,flip_angle,ETL,field_strength,scanner_name,scanner_model,slice_thickness,slice_spacing,pixel_spacing,rows,columns,acquisition_type,imaging_frequency,pixel_bandwidth,acquisition_matrix,SAR,percent_phase_FOV,percent_sampling,receive_coil
BACPAC_MDAI_t1sag_023,YkHJhdPAoL,75J7ITsLkW,1.3.6.1.4.1.20319.227560727460099300023534241017731357428,1.3.6.1.4.1.20319.221087568742653321421428832991626143074,1.3.6.1.4.1.20319.72915936643522625566197560692047820894,t1_tse_sag,MR LUMBAR SPINE WITH AND WITHOUT CONTRAST,9.3,518.0,140.0,3,3.0,UCSF-ZMOB3T,Verio,3.0,3.0,"['0.75', '0.75']",320,320,2D,123.217971,260.0,"[320, 0, 0, 224]",2.5317550529153,100.0,70.0,

*************************************************************

Spine UH3 T1sag
---------------

ML metadata
-----------
{
    "BACPAC0206004_01-20230517": {
        "subject_id": "BACPAC0206004_01-20230517",
        "image_nifti": "/data/mskdata/standardized/BACPAC_UH3/lumbar_spine/T1_sag/nifti/imgs/BACPAC0206004_01-20230517.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/BACPAC_UH3/lumbar_spine/T1_sag/nifti/masks/BACPAC0206004_01-20230517.nii.gz",
        "Sex": "F",
        "Age": "043Y",
        "Weight": 62.5957,
        "Dataset": "BACPAC_UH3_T1sag",
        "Anatomy": "lumbar_spine",
        "num_slices": 26,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "t1_sag",
        "Split": "train"
    },

Stats metadata
--------------
{
    "BACPAC0101001_01-20210818": {
        "Dataset": "BACPAC_UH3_T1sag",
        "Anatomy": "lumbar_spine",
        "slices": {
            "E501S11I1": {
                "PID": "VOLUNTEER",
                "AccessionNumber": "VOLUNTEER",
                "SOPInstanceUID": "1.2.840.113619.2.475.10795885.10187951.103239.1629297041.143",
                "StudyInstanceUID": "1.2.840.113619.6.475.31189100845801587676651809805904706429",
                "SeriesInstanceUID": "1.2.840.113619.2.475.10795885.10187951.107425.1629296987.247",
                "series_desc": "SAG T1 MSK",
                "TE": 8.088,
                "TR": 860.0,
                "flip_angle": 115.0,
                "ETL": 3,
                "field_strength": 3.0,
                "receive_coil": "60PA",
                "scanner_name": "UCSFCBMR7",
                "scanner_model": "SIGNA Premier",
                "slice_thickness": 3.0,
                "slice_spacing": 3.0,
                "pixel_spacing": [
                    0.5078,
                    0.5078
                ],
                "rows": 512,
                "columns": 512,
                "instanceNumber": 1
            },

Metadata Summary csv
--------------------
None

*************************************************************

OAI Thigh
----------

ML metadata
-----------
{
    "OAI_9021102": {
        "subject_id": "OAI_9021102",
        "image_nifti": "/data/mskdata/standardized/OAI/thigh/T1_ax/nifti/imgs/OAI_9021102.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/OAI/thigh/T1_ax/nifti/masks/OAI_9021102.nii.gz",
        "Sex": "F",
        "Age": 70,
        "Weight": 75.1,
        "Dataset": "OAI_thigh_muscle",
        "Anatomy": "thigh",
        "num_slices": 15,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "t1_ax",
        "Split": "train"
    },

Stats metadata
--------------
{
    "OAI_9021102": {
        "Dataset": "OAI_thigh_muscle",
        "Anatomy": "thigh",
        "slices": {
            "001": {
                "AccessionNumber": "016610439902",
                "SOPInstanceUID": "1.2.826.0.1.3680043.2.429.0.166104399.1123006545.1.2.1",
                "StudyInstanceUID": "1.3.12.2.1107.5.2.13.20579.4.0.1275405741862462",
                "SeriesInstanceUID": "1.3.12.2.1107.5.2.13.20579.4.0.1281179731323033",
                "series_desc": "AX_T1_THIGH",
                "study_desc": "OAI^MR^ENROLLMENT^THIGH",
                "TE": 10.0,
                "TR": 600.0,
                "flip_angle": 90.0,
                "ETL": 1,
                "field_strength": 2.89362,
                "scanner_name": "",
                "scanner_model": "Trio",
                "slice_thickness": 5.0,
                "slice_spacing": 5.0,
                "pixel_spacing": [
                    0.9765625,
                    0.9765625
                ],
                "rows": 256,
                "columns": 512,
                "instanceNumber": 1,
                "slice_location": 75.0,
                "image_position_patient": [
                    -233.53511,
                    -125.0,
                    75.0
                ]
            },

Metadata Summary csv
--------------------
,AccessionNumber,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,study_desc,TE,TR,flip_angle,ETL,field_strength,scanner_name,scanner_model,slice_thickness,slice_spacing,pixel_spacing,rows,columns,instanceNumber,slice_location,image_position_patient,acquisition_type,imaging_frequency,pixel_bandwidth,acquisition_matrix,SAR,percent_phase_FOV,percent_sampling
OAI_9021102,016610439902,1.2.826.0.1.3680043.2.429.0.166104399.1123006545.1.2.1,1.3.12.2.1107.5.2.13.20579.4.0.1275405741862462,1.3.12.2.1107.5.2.13.20579.4.0.1281179731323033,AX_T1_THIGH,OAI^MR^ENROLLMENT^THIGH,10.0,600.0,90.0,1,2.89362,,Trio,5.0,5.0,"['0.9765625', '0.9765625']",256,512,1,75.0,"['-233.53511', '-125.0', '75.0']",2D,123.25636,200.0,"[512, 0, 0, 256]",0.036723513,50.0,100.0

*************************************************************


K2S Knee
---------

ML metadata
-----------
{
    "TBrecon-01-02-00729": {
        "subject_id": "TBrecon-01-02-00729",
        "image_nifti": "/data/mskdata/standardized/TBrecon/knee/3D_CUBE_8x_undersampled/nifti/imgs/TBrecon-01-02-00729.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/TBrecon/knee/3D_CUBE_8x_undersampled/nifti/masks/TBrecon-01-02-00729.nii.gz",
        "Sex": "F",
        "Age": 54,
        "Weight": 72.57,
        "Dataset": "K2S",
        "Anatomy": "knee",
        "num_slices": 196,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "8x_undersampled_3D_CUBE",
        "Split": "train"
    },

Stats metadata
--------------
{
    "TBrecon-01-02-00729": {
        "Dataset": "K2S",
        "Anatomy": "knee",
        "slices": {
            "IM00001": {
                "PID": "99e43803-232c-48f2-a6ef-5d11f1dc0329",
                "SOPInstanceUID": "2.25.147292114357647248133757413990966220401",
                "StudyInstanceUID": "2.25.249065165528634672970561180003885074597",
                "SeriesInstanceUID": "2.25.26488415972361407524735794479457041430",
                "series_desc": "TBrecon-01-02-00729",
                "slice_thickness": 0.6,
                "pixel_spacing": [
                    0.293,
                    0.293
                ],
                "rows": 512,
                "columns": 512,
                "instanceNumber": 1
            },

Metadata Summary csv
--------------------
,PID,SOPInstanceUID,StudyInstanceUID,SeriesInstanceUID,series_desc,slice_thickness,pixel_spacing,rows,columns,instanceNumber
TBrecon-01-02-00729,99e43803-232c-48f2-a6ef-5d11f1dc0329,2.25.327263654751875922380413545908260284937,2.25.249065165528634672970561180003885074597,2.25.26488415972361407524735794479457041430,TBrecon-01-02-00729,0.6,"['0.293', '0.293']",512,512,196

*************************************************************


DL50 T2ax
----------

ML metadata
-----------
{
    "DLrecon_E1482": {
        "subject_id": "DLrecon_E1482",
        "image_nifti": "/data/mskdata/standardized/BACPAC/lumbar_spine/T2_ax/DL_50/nifti/imgs/DLrecon_E1482.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/BACPAC/lumbar_spine/T2_ax/DL_50/nifti/masks/DLrecon_E1482.nii.gz",
        "Sex": "F",
        "Age": "047Y",
        "Weight": 81.65,
        "Dataset": "BACPAC_MDai_T2_ax",
        "Anatomy": "lumbar_spine",
        "num_slices": 44,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "t2_ax",
        "Split": "train"
    },

Stats metadata
--------------
{
    "DLrecon_E1451": {
        "Dataset": "BACPAC_MDai_T2_ax",
        "Anatomy": "lumbar_spine",
        "slices": {
            "Image_00056": {
                "PID": "55879001",
                "AccessionNumber": "10020069531",
                "SOPInstanceUID": "1.2.840.113619.2.156.296502784.1.1589060379.448700",
                "StudyInstanceUID": "1.2.124.113532.80.22185.43466.20200416.105330.65403408",
                "SeriesInstanceUID": "1.2.840.113619.2.156.296502784.1.1589060379.448641",
                "series_desc": "NOT DIAGNOSTIC: FAST_AX T2_NEX0.5",
                "study_desc": "MR LUMBAR SPINE WITHOUT CONTRAST",
                "TE": 56.168,
                "TR": 5910.0,
                "flip_angle": 115.0,
                "ETL": 16,
                "field_strength": 3.0,
                "receive_coil": "32PA",
                "scanner_name": "UCSF-PCMR1",
                "scanner_model": "Orchestra SDK",
                "slice_thickness": 3.0,
                "slice_spacing": 3.0,
                "pixel_spacing": [
                    0.3516,
                    0.3516
                ],
                "rows": 512,
                "columns": 512,
                "instanceNumber": 57
            },

Metadata Summary csv
--------------------
None

*************************************************************


DL50 T1sag
-----------

ML metadata
-----------
{
    "DLrecon_E1482": {
        "subject_id": "DLrecon_E1482",
        "image_nifti": "/data/mskdata/standardized/BACPAC/lumbar_spine/T1_sag/DL_50/nifti/imgs/DLrecon_E1482.nii.gz",
        "mask_nifti": "/data/mskdata/standardized/BACPAC/lumbar_spine/T1_sag/DL_50/nifti/masks/DLrecon_E1482.nii.gz",
        "Sex": "F",
        "Age": "047Y",
        "Weight": 81.65,
        "Dataset": "BACPAC_MDai_T1_sag",
        "Anatomy": "lumbar_spine",
        "num_slices": 22,
        "mask_labels": {},
        "field_strength": "3.0",
        "mri_sequence": "t1_sag",
        "Split": "train"
    },

Stats metadata
--------------
{
    "DLrecon_E1482": {
        "Dataset": "BACPAC_MDai_T1_sag",
        "Anatomy": "lumbar_spine",
        "slices": {
            "Image_00020": {
                "PID": "07381158",
                "AccessionNumber": "10020077699",
                "SOPInstanceUID": "1.2.840.113619.2.156.296492800.1.1589399845.983477",
                "StudyInstanceUID": "1.2.124.113532.80.22185.43466.20200422.133806.6912571",
                "SeriesInstanceUID": "1.2.840.113619.2.156.296492800.1.1589399845.983456",
                "series_desc": "NOT DIAGNOSTIC: FAST_SAG T1_NEX0.5",
                "study_desc": "MR LUMBAR SPINE WITHOUT CONTRAST",
                "TE": 7.736,
                "TR": 870.0,
                "flip_angle": 105.0,
                "ETL": 4,
                "field_strength": 3.0,
                "receive_coil": "32PA",
                "scanner_name": "UCSF-PCMR1",
                "scanner_model": "Orchestra SDK",
                "slice_thickness": 3.0,
                "slice_spacing": 3.0,
                "pixel_spacing": [
                    0.5078,
                    0.5078
                ],
                "rows": 512,
                "columns": 512
            },

Metadata Summary csv
--------------------
None

*************************************************************