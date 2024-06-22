
from albumentations import (
    Compose, OneOf, RandomBrightnessContrast, RandomGamma, ElasticTransform,
    GridDistortion, OpticalDistortion, RandomSizedCrop, ToFloat, RandomCrop, 
    HorizontalFlip, VerticalFlip, Resize
)

def get_transform(transform_name, **kwargs):
    if transform_name == "RandomCrop":
        return RandomCrop(**kwargs)
    elif transform_name == "HorizontalFlip":
        return HorizontalFlip(**kwargs)
    elif transform_name == "VerticalFlip":
        return VerticalFlip(**kwargs)
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
    return Compose(transforms)





# augmentation_pipeline:
#   train:
#     - transform: OneOf
#       args: 
#         transforms:
#           - transform: RandomBrightnessContrast
#             args: {}
#           - transform: RandomGamma
#             args: {}
#       p: 0.3
#     # Other transforms as needed


# augmentation_pipeline:
#   train:
#     - transform: RandomCrop
#       args: 
#         height: 256
#         width: 256
#     - transform: HorizontalFlip
#       args: {}
#   val:
#     - transform: Resize
#       args:
#         height: 256
#         width: 256

# transform = A.Compose([
#     # A.RandomCrop(width=450, height=450),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ], bbox_params=A.BboxParams(format='voc'))
# transformed = transform(image=image, bboxes=bboxes)
# transformed_image = transformed['image']
# transformed_bboxes = transformed['bboxes']


