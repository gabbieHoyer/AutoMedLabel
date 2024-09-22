
from albumentations import (
    Compose, OneOf, RandomBrightnessContrast, RandomGamma, ElasticTransform,
    GridDistortion, OpticalDistortion, RandomSizedCrop, ToFloat, RandomCrop, 
    HorizontalFlip, VerticalFlip, Resize, RandomRotate90
)

# from monai.transforms import (
#     RandFlipd, RandAffined, RandGaussianNoised, RandRotated, RandZoomd
# )


# --------- Albumentataions Only ---------- #

def get_transform(transform_name, **kwargs):
    if transform_name == "RandomCrop":
        return RandomCrop(**kwargs)
    elif transform_name == "HorizontalFlip":
        return HorizontalFlip(**kwargs)
    elif transform_name == "VerticalFlip":
        return VerticalFlip(**kwargs)
    elif transform_name == "RandomRotate90":
        return RandomRotate90(**kwargs)
    elif transform_name == "Resize":
        return Resize(**kwargs)
    elif transform_name == "RandomBrightnessContrast":
        return RandomBrightnessContrast(**kwargs)
    elif transform_name == "RandomGamma":
        return RandomGamma(**kwargs)
    elif transform_name == "ElasticTransform":
        return ElasticTransform(**kwargs)
    elif transform_name == "GridDistortion":
        return GridDistortion(**kwargs)
    elif transform_name == "OpticalDistortion":
        return OpticalDistortion(**kwargs)
    elif transform_name == "RandomSizedCrop":
        return RandomSizedCrop(**kwargs)
    elif transform_name == "ToFloat":
        return ToFloat(**kwargs)
    else:
        raise ValueError(f"Unsupported transform: {transform_name}")
    

def build_augmentation_pipeline(config):
    if config is None:
        return None  # Explicitly return None if there's no config

    transforms = []
    for item in config:
        if item['transform'] == "OneOf":
            one_of_transforms = [get_transform(t['transform'], **t.get('args', {})) for t in item['args']['transforms']]
            transforms.append(OneOf(one_of_transforms, p=item.get('p', 1)))
        else:
            transforms.append(get_transform(item['transform'], **item.get('args', {})))
   
    # Turn off the shape check by setting is_check_shapes=False
    return Compose(transforms, is_check_shapes=False)





# ---------------- Monai variant in dev -------------------- #
# def apply_augmentations(augmentation_pipeline, img, mask, pipeline_library):
#     """
#     Apply the augmentation pipeline based on the specified library.
    
#     :param augmentation_pipeline: The composed augmentation pipeline.
#     :param img: The image data.
#     :param mask: The mask data.
#     :param pipeline_library: Which library is being used ('albumentations' or 'monai').
#     :return: Augmented image and mask.
#     """
#     if pipeline_library == 'albumentations':
#         # Albumentations expects keyword arguments
#         augmented = augmentation_pipeline(image=img, mask=mask)
#         return augmented['image'], augmented['mask']
    
#     elif pipeline_library == 'monai':
#         # MONAI expects a dictionary input
#         augmented = augmentation_pipeline({"image": img, "mask": mask})
#         return augmented['image'], augmented['mask']

#     else:
#         raise ValueError(f"Unsupported library: {pipeline_library}")


# def build_augmentation_pipeline(config):
#     """
#     Builds the augmentation pipeline with both albumentations and monai transforms,
#     using conditional checks to separate the logic.
    
#     :param config: A list of dictionaries specifying the transformations and arguments.
#     :return: A composed transform pipeline.
#     """
#     if config is None:
#         return None  # Explicitly return None if there's no config

#     import pdb; pdb.set_trace()

#     # Set the library for the entire pipeline (default to albumentations if not specified)
#     pipeline_library = config.get('library', 'albumentations')

#     transforms = []

#     if pipeline_library == "albumentations":
#         from albumentations import Compose, OneOf

#         for item in config['transforms']:
#             if item['transform'] == "OneOf":
#                 one_of_transforms = [get_transform(t['transform'], library="albumentations", **t.get('args', {})) for t in item['args']['transforms']]
#                 transforms.append(OneOf(one_of_transforms, p=item.get('p', 1)))
#             else:
#                 transforms.append(get_transform(item['transform'], library="albumentations", **item.get('args', {})))

#         # Return Compose for albumentations
#         return Compose(transforms)

#     elif pipeline_library == "monai":
#         from monai.transforms import Compose, OneOf

#         for item in config['transforms']:
#             if item['transform'] == "OneOf":
#                 one_of_transforms = [get_transform(t['transform'], library="monai", **t.get('args', {})) for t in item['args']['transforms']]
#                 weights = item.get('args').get('weights')  # Get weights if provided in the config
#                 transforms.append(OneOf(one_of_transforms, weights=weights))
#             else:
#                 transforms.append(get_transform(item['transform'], library="monai", **item.get('args', {})))

#         # Return Compose for monai
#         return Compose(transforms)

#     else:
#         raise ValueError(f"Unsupported library: {pipeline_library}")



# def get_transform(transform_name, library='albumentations', **kwargs):
#     """
#     Handles transforms for both albumentations and monai libraries.
    
#     :param transform_name: The name of the transform.
#     :param library: Specifies whether the transform is from albumentations or monai.
#     :param kwargs: Additional arguments for the transform.
#     :return: The corresponding transform.
#     """
#     if library == 'albumentations':
#         if transform_name == "RandomCrop":
#             return RandomCrop(**kwargs)
#         elif transform_name == "HorizontalFlip":
#             return HorizontalFlip(**kwargs)
#         elif transform_name == "VerticalFlip":
#             return VerticalFlip(**kwargs)
#         elif transform_name == "RandomRotate90":
#             return RandomRotate90(**kwargs)
#         elif transform_name == "Resize":
#             return Resize(**kwargs)
#         elif transform_name == "RandomBrightnessContrast":
#             return RandomBrightnessContrast(**kwargs)
#         elif transform_name == "RandomGamma":
#             return RandomGamma(**kwargs)
#         elif transform_name == "ElasticTransform":
#             return ElasticTransform(**kwargs)
#         elif transform_name == "GridDistortion":
#             return GridDistortion(**kwargs)
#         elif transform_name == "OpticalDistortion":
#             return OpticalDistortion(**kwargs)
#         elif transform_name == "RandomSizedCrop":
#             return RandomSizedCrop(**kwargs)
#         elif transform_name == "ToFloat":
#             return ToFloat(**kwargs)
#         else:
#             raise ValueError(f"Unsupported albumentations transform: {transform_name}")

#     elif library == 'monai':
#         if transform_name == "RandFlipd":
#             return RandFlipd(**kwargs)
#         elif transform_name == "RandAffined":
#             return RandAffined(**kwargs)
#         elif transform_name == "RandGaussianNoised":
#             return RandGaussianNoised(**kwargs)
#         elif transform_name == "RandRotated":
#             return RandRotated(**kwargs)
#         elif transform_name == "RandZoomd":
#             return RandZoomd(**kwargs)
#         else:
#             raise ValueError(f"Unsupported monai transform: {transform_name}")

#     else:
#         raise ValueError(f"Unsupported library: {library}")












