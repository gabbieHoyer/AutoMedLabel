import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import random



def map_labels_to_colors(pred_mask, mask_labels):
    # Define a colormap that can provide a distinct color for each class
    color_map = plt.get_cmap('tab20', len(mask_labels))  # 'tab20' has 20 distinct colors

    # Create an empty RGBA image
    colored_mask = np.zeros((*pred_mask.shape, 4), dtype=np.float32)  # Initialize with zeros

    # Map each label to a consistent color from the colormap
    for label_value, label_name in mask_labels.items():
        if label_value == 0:  # Skip the background
            continue
        mask = (pred_mask == label_value)
        color = color_map(label_value / len(mask_labels))  # Get consistent RGBA color
        colored_mask[mask] = color  # Apply color where the label matches

    return colored_mask

# --------------------------------------------------------------------------- #

def visualize_input(image, gt_mask, box, pred_mask, label_id, image_name, model_save_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with three subplots

    # Original image
    # Assuming image is in [C, H, W] format and is an RGB image
    ax[0].imshow(image.permute(1, 2, 0), cmap='gray')  # Assuming image is in [C, H, W] format
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Ground truth mask
    ax[1].imshow(gt_mask, cmap='rainbow')
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')

    # Draw bounding boxes on the original image
    rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)

    # Predicted mask
    ax[2].imshow(pred_mask, cmap='rainbow')
    ax[2].set_title(f'Predicted Mask: {label_id}')
    ax[2].axis('off')

    figure_file_path = os.path.join(model_save_path, 'qual_check', f"{image_name}_input_viz.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path) 
    plt.close(fig)  # Close the figure to free memory


def visualize_predictions(image, gt_mask, pred_mask, boxes, mask_labels, image_name, model_save_path):  # pred_mask_bin, label_id,
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with three subplots

    # Original image
    # Assuming image is in [C, H, W] format and is an RGB image
    ax[0].imshow(image.permute(1, 2, 0))  # Assuming image is in [C, H, W] format
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Ground truth mask
    ax[1].imshow(gt_mask, cmap='gray')
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')

    # Prediction overlay
    ax[2].imshow(image.permute(1, 2, 0))  # Original image
    colored_pred_mask = map_labels_to_colors(pred_mask.cpu().numpy(), mask_labels)
    ax[2].imshow(colored_pred_mask, alpha=0.5)  # Overlay the colored mask
    ax[2].set_title('Prediction Overlay')
    ax[2].axis('off')

    # Draw bounding boxes on the original image
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

    figure_file_path = os.path.join(model_save_path, 'test_eval', f"{image_name}_pred_viz.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path)  # Save the plot to a file
    plt.close(fig)  # Close the figure to free memory
    

# ----------------------------------------------------------------------------------------- #

def full_scale_visualize_input(output_dir, image, gt_mask, boxes, label_id, image_name):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure with two subplots

    # Original image
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())  # Assuming image is in [C, H, W] format
    ax[0].set_title(f'Original Image with Boxes for Label ID: {label_id}')
    ax[0].axis('off')

    # Draw bounding boxes on the original image
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

    # Ground truth mask
    ax[1].imshow(gt_mask.cpu().numpy(), cmap='gray')
    ax[1].set_title(f'Ground Truth Mask for Label ID: {label_id}')
    ax[1].axis('off')

    figure_file_path = os.path.join(output_dir, 'qual_check', f"QC_visualization_{image_name}_label_{label_id}.png")
    os.makedirs(os.path.dirname(figure_file_path), exist_ok=True)

    plt.savefig(figure_file_path)
    plt.close(fig)  # Close the figure to free memory

# ----------------------------------------------------------------------------------------- #

def quality_check(test_loader, output_dir):
    # Determine the number of batches
    num_batches = len(test_loader)

    # Randomly select a batch index
    random_batch_idx = random.randint(0, num_batches - 1)

    for batch_idx, batch in enumerate(test_loader):
        if batch_idx != random_batch_idx:
            continue

        img_names = batch['img_name']
        images, gt2D, boxes, label_ids = batch['image'], batch['gt2D'], batch['boxes'], batch['label_ids']

        # Iterate over each item in the batch
        for i in range(images.size(0)):
            image = images[i]
            img_name = img_names[i]

            # Group boxes and masks by label_id
            label_boxes_masks = {}
            for j in range(len(label_ids[i])):
                label_id = label_ids[i][j].item()
                if label_id not in label_boxes_masks:
                    # Only store the mask once for each label_id
                    label_boxes_masks[label_id] = {'boxes': [], 'mask': (gt2D[i, 0] == label_id).float()}
                label_boxes_masks[label_id]['boxes'].append(boxes[i][j])

            # Visualize each label_id's boxes and masks
            for label_id, data in label_boxes_masks.items():
                full_scale_visualize_input(output_dir, image, data['mask'], data['boxes'], label_id, img_name)


# def quality_check(test_loader, output_dir):
#     for batch_idx, batch in enumerate(test_loader):
#         if batch_idx == 1:  # Only process the first batch
#             break

#         img_names = batch['img_name']
#         images, gt2D, boxes, label_ids = batch['image'], batch['gt2D'], batch['boxes'], batch['label_ids']

#         # Iterate over each item in the batch
#         for i in range(images.size(0)):
#             image = images[i]
#             img_name = img_names[i]

#             # Group boxes and masks by label_id
#             label_boxes_masks = {}
#             for j in range(len(label_ids[i])):
#                 label_id = label_ids[i][j].item()
#                 if label_id not in label_boxes_masks:
#                     # Only store the mask once for each label_id
#                     label_boxes_masks[label_id] = {'boxes': [], 'mask': (gt2D[i, 0] == label_id).float()}
#                 label_boxes_masks[label_id]['boxes'].append(boxes[i][j])

#             # Visualize each label_id's boxes and masks
#             for label_id, data in label_boxes_masks.items():
#                 full_scale_visualize_input(output_dir, image, data['mask'], data['boxes'], label_id, img_name)



# def apply_binary_mask_color(pred_mask_bin, mask_labels, label_id):
#     # Define a colormap that can provide a distinct color for each class
#     color_map = plt.get_cmap('tab20b', len(mask_labels) - 1)  # 'tab20b' has 20 distinct colors, excluding background

#     # Initialize the RGBA color image
#     binary_mask_rgba = np.zeros((*pred_mask_bin.shape, 4), dtype=np.float32)  # Initialize with zeros

#     # Get the color for the label_id
#     color_idx = list(mask_labels.keys()).index(label_id)  # Get the index of the label in the dictionary
#     mask_color = color_map(color_idx / (len(mask_labels) - 1))  # Use index to get a consistent color

#     # Set the RGBA color where the binary mask is 1
#     binary_mask_rgba[pred_mask_bin == 1, :3] = mask_color[:3]  # Apply color
#     binary_mask_rgba[pred_mask_bin == 1, 3] = 0.5  # Set alpha to 0.5 for the binary mask

#     return binary_mask_rgba
