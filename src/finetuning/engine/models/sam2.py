import torch
import torch.nn as nn
import torch.nn.functional as F

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)
from src.sam2.utils.transforms import SAM2Transforms

class finetunedSAM2(nn.Module):
    def __init__(self, model, config, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0):
        super().__init__()
        self.sam2_model = model
        self.resize_masks = config.get("resize_masks", False)
        
        # Freeze components according to configuration.
        if not config['prompt_encoder']:
            for param in self.sam2_model.sam_prompt_encoder.parameters():
                param.requires_grad = False

        if not config['image_encoder']:
            for name, param in self.sam2_model.named_parameters():
                param.requires_grad = False
            for name, param in self.sam2_model.named_parameters():
                if 'sam_mask_decoder' in name:
                    param.requires_grad = True

        # If resizing is enabled, initialize the transform helper.
        if self.resize_masks:
            self._transforms = SAM2Transforms(
                resolution=self.sam2_model.image_size,
                mask_threshold=mask_threshold,
                max_hole_area=max_hole_area,
                max_sprinkle_area=max_sprinkle_area,
            )
        else:
            self._transforms = None

        # Spatial sizes for backbone feature maps.
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

    def forward(self, image, box):
        """
        image: Tensor of shape (B, 3, H, W)
        box: Tensor or array of shape (B, 2, 2)
        """
        features = self._image_encoder(image)
        img_embed = features["image_embed"]
        high_res_features = features["high_res_feats"]

        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)
            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )

        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = \
            self.sam2_model.sam_mask_decoder(
                image_embeddings=img_embed,
                image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

        if self.resize_masks and self._transforms is not None:
            output_masks_logits = self._transforms.postprocess_masks(
                low_res_masks_logits, orig_hw=(image.size(2), image.size(3))
            )
        else:
            output_masks_logits = low_res_masks_logits

        return output_masks_logits

    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return features

