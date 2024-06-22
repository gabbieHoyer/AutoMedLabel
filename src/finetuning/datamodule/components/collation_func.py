
import torch

def collate_fn(batch, use_biomarkers):
    images = [item['image'] for item in batch]
    gt2D = [item['gt2D'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    label_ids = [item['label_ids'] for item in batch]
    img_names = [item['img_name'] for item in batch]

    images = torch.stack(images)
    gt2D = torch.stack(gt2D)

    if use_biomarkers:
        t1rho_maps = torch.stack([item['t1rho'] for item in batch])
        t2_maps = torch.stack([item['t2'] for item in batch])

    max_num_boxes = max(box.shape[0] for box in boxes)

    padded_boxes = []
    padded_labels = []

    for box, label in zip(boxes, label_ids):
        num_boxes = box.shape[0]
        padded_box = torch.zeros((max_num_boxes, 4))
        padded_label = torch.zeros((max_num_boxes,))
        if num_boxes > 0:
            padded_box[:num_boxes, :] = box
            padded_label[:num_boxes] = label
        padded_boxes.append(padded_box)
        padded_labels.append(padded_label)

    padded_boxes = torch.stack(padded_boxes)
    padded_labels = torch.stack(padded_labels)

    batch_data = {
        'image': images,
        'gt2D': gt2D,
        'boxes': padded_boxes,
        'label_ids': padded_labels,
        'img_name': img_names
    }

    if use_biomarkers:
        batch_data.update({
            't1rho': t1rho_maps,
            't2': t2_maps
        })

    return batch_data

# def collate_fn(batch):
#     images = [item['image'] for item in batch]
#     gt2D = [item['gt2D'] for item in batch]
#     boxes = [item['boxes'] for item in batch]
#     label_ids = [item['label_ids'] for item in batch]
#     img_names = [item['img_name'] for item in batch]

#     images = torch.stack(images)
#     gt2D = torch.stack(gt2D)
#     max_num_boxes = max([box.shape[0] for box in boxes])

#     padded_boxes = []
#     padded_labels = []

#     for box, label in zip(boxes, label_ids):
#         num_boxes = box.shape[0]
#         padded_box = torch.zeros((max_num_boxes, 4))
#         padded_label = torch.zeros((max_num_boxes,))
#         if num_boxes > 0:
#             padded_box[:num_boxes, :] = box
#             padded_label[:num_boxes] = label
#         padded_boxes.append(padded_box)
#         padded_labels.append(padded_label)

#     padded_boxes = torch.stack(padded_boxes)
#     padded_labels = torch.stack(padded_labels)

#     return {
#         'image': images,
#         'gt2D': gt2D,
#         'boxes': padded_boxes,
#         'label_ids': padded_labels,
#         'img_name': img_names
#     }


# def collate_maps_fn(batch):
#     images = [item['image'] for item in batch]
#     gt2D = [item['gt2D'] for item in batch]
#     boxes = [item['boxes'] for item in batch]
#     label_ids = [item['label_ids'] for item in batch]
#     img_names = [item['img_name'] for item in batch]

#     # ------------------------------
#     t1rho_maps = [item['t1rho'] for item in batch]
#     t2_maps = [item['t2'] for item in batch]
#     # ------------------------------

#     images = torch.stack(images)
#     gt2D = torch.stack(gt2D)

#     # ------------------------------
#     t1rho_maps = torch.stack(t1rho_maps)
#     t2_maps = torch.stack(t2_maps)
#     # ------------------------------

#     max_num_boxes = max([box.shape[0] for box in boxes])

#     padded_boxes = []
#     padded_labels = []

#     for box, label in zip(boxes, label_ids):
#         num_boxes = box.shape[0]
#         padded_box = torch.zeros((max_num_boxes, 4))
#         padded_label = torch.zeros((max_num_boxes,))
#         if num_boxes > 0:
#             padded_box[:num_boxes, :] = box
#             padded_label[:num_boxes] = label
#         padded_boxes.append(padded_box)
#         padded_labels.append(padded_label)

#     padded_boxes = torch.stack(padded_boxes)
#     padded_labels = torch.stack(padded_labels)

#     return {
#         'image': images,
#         'gt2D': gt2D,
#         'boxes': padded_boxes,
#         'label_ids': padded_labels,
#         'img_name': img_names,
#         't1rho': t1rho_maps,
#         't2': t2_maps
#     }
