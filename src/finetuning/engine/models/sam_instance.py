import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide

from typing import Any, Optional, Tuple, Type

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

# from src.finetuning.engine.models.components.prompt_encoder import PromptEncoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
# from src.finetuning.engine.models.components.prompt_encoder import PromptEncoder
# 


# class multiInstancePromptEncoder(PromptEncoder):
#     def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
#         """Embeds box prompts with a custom modification."""
#         boxes = boxes + 0.5  # Shift to center of pixel
#         coords = boxes.reshape(-1, 2, 2)
#         corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
#         corner_embedding[:, 0, :] += self.point_embeddings[2].weight
#         corner_embedding[:, 1, :] += self.point_embeddings[3].weight

#         import pdb; pdb.set_trace()
#         # Custom return statement
#         return corner_embedding.reshape(1, -1, self.embed_dim)
    
# from src.finetuning.engine.models.components.prompt_encoder import PromptEncoder

# class multiInstancePromptEncoder(PromptEncoder):
#     def __init__(self, *args, **kwargs):
#         super(multiInstancePromptEncoder, self).__init__(*args, **kwargs)

#     def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
#         """Custom embed boxes method with altered return shape."""
#         boxes = boxes + 0.5  # Shift to center of pixel
#         coords = boxes.reshape(-1, 2, 2)
#         corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
#         corner_embedding[:, 0, :] += self.point_embeddings[2].weight
#         corner_embedding[:, 1, :] += self.point_embeddings[3].weight

#         import pdb; pdb.set_trace()

#         return corner_embedding.reshape(1, -1, self.embed_dim)  # Custom return statement


class multiInstancePromptEncoder(PromptEncoder):
    def forward(self, points: Optional[Tuple[torch.Tensor, torch.Tensor]], boxes: Optional[torch.Tensor], masks: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Customized forward method to alter box embeddings handling.
        """
        import pdb; pdb.set_trace()
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        import pdb; pdb.set_trace()
        if boxes is not None:
            # Custom handling of box embeddings
            box_embeddings = self._embed_boxes(boxes)

            # Example customization: reshape or modify box_embeddings here before concatenation
            box_embeddings = box_embeddings.reshape(1, -1, self.embed_dim)  # Example modification

            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class finetunedSAM(nn.Module):
    def __init__(self, image_encoder, prompt_encoder, mask_decoder, config):
        super().__init__()
        self.prompt_encoder = prompt_encoder 
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        
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

        import pdb; pdb.set_trace()

    def forward(self, image, box):
        # Compute image embedding
        image_embedding = self.image_encoder(image) 

        # Convert box to tensor
        box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)

        import pdb; pdb.set_trace()

        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :] 

        # Check if box_torch is empty
        # if box_torch.numel() == 0:
        #     box_torch = None
        # else:
        #     # box_torch remains unchanged, retains its original shape
        #     pass

        # # input_boxes = np.array(boxes) if len(boxes) > 0 else None

        # resize_transform = ResizeLongestSide(1024) #sam.image_encoder.img_size)

        # transformed_box_torch = resize_transform.apply_boxes_torch(box_torch, image.shape[:2])

        # Generate sparse and dense embeddings
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

        # import pdb; pdb.set_trace()

        # sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #     points=None,
        #     boxes=transformed_box_torch,
        #     masks=None,
        # )

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
