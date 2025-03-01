# Functions to get/load and display images

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap

# -------------------- Color Tools --------------------
class ColorInfo():
    """
    TODO
    """
    # ---- FUNCTIONS NOT DEPENDENT ON DATA EXTENSION -----

    def set_unique_labels(self, mask):
        return [int(label) for label in np.unique(mask[mask != 0])]
    
    def set_unique_bbox_labels(self, volume_bbox_list):
        """
        - volume_bbox_list: [(slice, slice_bbox_list),...], where slice_bbox_list=[(label_id, bbox), ...]
        """
        if not isinstance(volume_bbox_list[0], tuple) and not isinstance(volume_bbox_list[0], list):
            volume_bbox_list = (0, volume_bbox_list)
        
        # Process each slice
        label_id_list = []
        for slice_idx, slice_bbox_list in volume_bbox_list:
            # Process each label and get the bounding boxes
            for label_id, bbox in slice_bbox_list:
                if label_id not in label_id_list:
                    label_id_list.append(label_id)

        return np.array(label_id_list)

    def set_image_clim(self, image):
        image_norm = 'percentile'
        # can do percentiles here
        if image_norm == 'percentile':
            return [np.percentile(image[:], 2),np.percentile(image[:], 98)]
        else:
            return [np.min(image[:]),np.max(image[:])]
    
    def set_mask_clim(self, mask):
        """
        Adjust 'clim' based on 'unique_labels' before calling 'adjust_cmap_for_labels'
        """
        unique_labels = self.set_unique_labels(mask)
        
        if not unique_labels:
            clim = [0,1]
        elif unique_labels == [0]:
            clim = [0,1]
        elif len(unique_labels) == 1:
            clim = [0, np.max(unique_labels)]
        elif len(unique_labels) > 1:
            clim = [min(unique_labels), max(unique_labels)]
        # didnt work for dhal or bacpac sag1
        return clim
    
    def set_bbox_colors(self, volume_bbox_list, cmap:str='rainbow'):
        """
        Generate colors from the colormap
        - volume_bbox_list: [(slice, slice_bbox_list),...], where slice_bbox_list=[(label_id, bbox), ...]
        """
        unique_labels = self.set_unique_bbox_labels(volume_bbox_list)
        
        # Get a color map
        colormap = cm.get_cmap(cmap)
        # Generate colors from the colormap
        label_colors = [colormap(i / len(unique_labels)) for i in range(len(unique_labels))]
    
        return dict(zip(unique_labels, label_colors))

    def adjust_cmap_for_labels(self, cmap, labels, clim):
        """
        Adjust a given colormap to fit the specified labels within the color limit (clim) range,
        ensuring each label has a consistent color across the volume.
        """
        base_cmap = plt.get_cmap(cmap)
        # If clim is provided and valid, use it to normalize
        if clim and clim[0] < clim[1]:
            norm = Normalize(vmin=clim[0], vmax=clim[1])
        else:
            # If only one label or clim not valid, adjust the range to ensure a valid normalization
            label_value = min(labels) if labels else 0  # Use 0 if labels are empty
            norm = Normalize(vmin=label_value - 0.1, vmax=label_value + 0.1)

        # Generate a new colormap from the base colormap using the normalized label values
        colors = base_cmap(np.linspace(0, 1, len(labels)))
        new_cmap = ListedColormap(colors)

        return new_cmap, norm

    def add_colorbar(self, fig, ax, labels_dict, cmap, norm, unique_labels):
        # Assuming background is already excluded from unique_labels and labels_dict

        # Calculate the boundaries and ticks using the unique labels
        boundaries = np.linspace(norm.vmin, norm.vmax, len(unique_labels) + 1)
        ticks = (boundaries[:-1] + boundaries[1:]) / 2  # get the mid-point of boundaries

        # Create the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy array for ScalarMappable
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, boundaries=boundaries, ticks=ticks)

        # Set the tick labels
        cbar.set_ticklabels([labels_dict[label] for label in unique_labels])

        cbar.ax.tick_params(labelsize=12)
        plt.setp(cbar.ax.get_xticklabels(), rotation=45, ha='right')
