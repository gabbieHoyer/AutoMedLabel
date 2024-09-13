import torch
import torch.nn as nn
import torch.nn.functional as F


class finetunedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder, config):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
        # Apply trainable configuration / Freeze components based on config
        if not config['prompt_encoder']:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        if not config['image_encoder']:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        if not config['mask_decoder']:
            for param in self.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, image, box):
        # Compute image embedding
        image_embedding = self.image_encoder(image) 

        # Convert box to tensor
        box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] 

        # Generate sparse and dense embeddings
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        # Generate low resolution masks
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding, 
            image_pe=self.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings,  
            dense_prompt_embeddings=dense_embeddings,  
            multimask_output=False,
        )
        # Interpolate to original image resolution
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks
    