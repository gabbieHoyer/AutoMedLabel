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
    def __init__(self, model, config):
        super().__init__()
       
        # medsam way
        self.sam2_model = model
        # # freeze prompt encoder
        # for param in self.sam2_model.sam_prompt_encoder.parameters():
        #     param.requires_grad = False

        # Apply trainable configuration / Freeze components based on config
        if not config['prompt_encoder']:
            for param in self.sam2_model.sam_prompt_encoder.parameters():
                param.requires_grad = False

        if not config['image_encoder']:
            for name, param in self.sam2_model.named_parameters():
                param.requires_grad = False

            for name, param in self.sam2_model.named_parameters():
                if 'sam_mask_decoder' in name:
                    param.requires_grad = True

        # mask decoder should always be finetuned in our case
        # if not config['mask_decoder']:
        #     for param in self.sam2_model.sam_mask_decoder.parameters():
        #         param.requires_grad = False

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [(256, 256),(128, 128),(64, 64),]

    def forward(self, image, box):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            # from sam2_image_predictor.py ~line 395
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return low_res_masks_logits
    
    def _image_encoder(self, input_image):
        # From sam2_image_predictor.py def set_image:

        backbone_out = self.sam2_model.forward_image(input_image)
        
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # backbone_out, vision_feats, vision_pos_embeds, feat_sizes = '''
        #NOTE: check if these feat_sizes are same or diff from bb_feat_sizes

        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features



class finetunedSAM2_1024(nn.Module):
    def __init__(self, model, config, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0):
        super().__init__()
       
        # medsam way
        self.sam2_model = model
        # # freeze prompt encoder
        # for param in self.sam2_model.sam_prompt_encoder.parameters():
        #     param.requires_grad = False

        # Apply trainable configuration / Freeze components based on config
        # if not config['prompt_encoder']:
        #     for param in self.sam2_model.sam_prompt_encoder.parameters():
        #         param.requires_grad = False

        # if not config['image_encoder']:
        #     for param in self.sam2_model.sam_image_encoder.parameters():
        #         param.requires_grad = False

        # if not config['mask_decoder']:
        #     for param in self.sam2_model.sam_mask_decoder.parameters():
        #         param.requires_grad = False

        if not config['prompt_encoder']:
            for param in self.sam2_model.sam_prompt_encoder.parameters():
                param.requires_grad = False

        if not config['image_encoder']:
            for name, param in self.sam2_model.named_parameters():
                param.requires_grad = False

            for name, param in self.sam2_model.named_parameters():
                if 'sam_mask_decoder' in name:
                    param.requires_grad = True

        self._transforms = SAM2Transforms(
            resolution=self.sam2_model.
            image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [(256, 256),(128, 128),(64, 64),]

    def forward(self, image, box):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            # from sam2_image_predictor.py ~line 395
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        resized_masks_logits = self._transforms.postprocess_masks(
            low_res_masks_logits, orig_hw=(image.size(2), image.size(3))
        )
        # resized_masks_logits = F.interpolate(
        #     low_res_masks_logits,size=(image.size(2), image.size(3)), 
        #     mode="bilinear", align_corners=False)

        return resized_masks_logits

        # resized_masks_logits = F.interpolate(
        #     low_res_masks_logits,size=(image.size(2), image.size(3)), 
        #     mode="bilinear", align_corners=False)

        # return resized_masks_logits
        
    
        # from sam2 codebase - they upscale masks 
        #     # Upscale the masks to the original image resolution
        # masks = self._transforms.postprocess_masks(
        #     low_res_masks, self._orig_hw[img_idx]
        # )
        # low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        # if not return_logits:
        #     masks = masks > self.mask_threshold

        # return masks, iou_predictions, low_res_masks
    
    def _image_encoder(self, input_image):
        # From sam2_image_predictor.py def set_image:

        backbone_out = self.sam2_model.forward_image(input_image)
        
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # backbone_out, vision_feats, vision_pos_embeds, feat_sizes = '''
        #NOTE: check if these feat_sizes are same or diff from bb_feat_sizes

        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        
        # bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features





    # def forward(self, image, box):
    #     image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

    #     with torch.no_grad():
    #         box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
    #         if len(box_torch.shape) == 2:
    #             box_torch = box_torch[:, None, :]  # (B, 1, 4)

    #         sparse_embeddings, dense_embeddings = self.prompt_encoder(
    #             points=None,
    #             boxes=box_torch,
    #             masks=None,
    #         )
    #     low_res_masks, _ = self.mask_decoder(
    #         image_embeddings=image_embedding,  # (B, 256, 64, 64)
    #         image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
    #         sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
    #         dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
    #         multimask_output=False,
    #     )
