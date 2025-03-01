
import os
import logging
from torch.optim import AdamW, SGD, RMSprop, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, 
    StepLR, 
    CosineAnnealingLR, 
    ExponentialLR
)

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)
logger = logging.getLogger(__name__)

# ---------------------- Select Segmentation Base Model ---------------------- #

def load_segmentation_model(trainable_cfg, model_config, weights_path, device):
    """
    Create and return the segmentation model.

    Args:
        trainable_cfg: Configuration for the trainable module.
        model_config: For SAM this is a model key (e.g. 'vit_b'); for SAM2 this is a YAML config file.
        weights_path: Path to the pretrained weights.
        device: The device for the model.

    Returns:
        The segmentation model.
    """
    if model_config.endswith('.yaml'):
        # Assume SAM2 if the .yaml config provided as per SAM2 system.
        from src.sam2.build_sam import build_sam2
        # from src.finetuning.engine.models.sam2 import finetunedSAM2
        from src.finetuning.engine.models import finetunedSAM2

        base_model = build_sam2(model_config, weights_path, device=device, apply_postprocessing=True)
        seg_model = finetunedSAM2( 
            model=base_model,
            config=trainable_cfg
        ).to(device)
    else:
        # Otherwise assume SAM-based model.
        # Here model_config acts as a key for the SAM registry.
        from src.segment_anything import sam_model_registry
        # from src.finetuning.engine.models.sam import finetunedSAM
        from src.finetuning.engine.models import finetunedSAM

        base_model = sam_model_registry[model_config](checkpoint=weights_path)
        seg_model = finetunedSAM( 
            image_encoder=base_model.image_encoder,
            mask_decoder=base_model.mask_decoder,
            prompt_encoder=base_model.prompt_encoder,
            config=trainable_cfg
        ).to(device)
    return seg_model

# ---------------------- Select Object Detection Base Model ---------------------- #

def load_detection_model(det_cfg):
    """
    Create and return a detection model based on the given configuration.

    Args:
        det_cfg (dict): Detection configuration. Expected keys include:
            - model_type: Type of detection model (e.g., 'YOLO', 'RTDETR').
            - model_path: Relative path to the model weights.
            - root (optional): Base directory for the model weights. If not provided, a global 'root' is used.

    Returns:
        An instantiated detection model.

    Raises:
        ValueError: If 'model_path' is not provided or if 'model_type' is unsupported.
    """
    # Determine base directory: use provided value or fall back to a global variable
    base_dir = det_cfg.get('root', root)
    model_path = det_cfg.get('model_path')
    
    if not model_path:
        raise ValueError("Detection configuration must include 'model_path'.")

    model_type = det_cfg.get('model_type')
    full_model_path = os.path.join(base_dir, model_path)

    if model_type == 'YOLO':
        from src.obj_detection.ultralytics import YOLO
        det_model = YOLO(full_model_path)
    elif model_type == 'RTDETR':
        from src.obj_detection.ultralytics import RTDETR
        det_model = RTDETR(full_model_path)
    else:
        raise ValueError(f"Unsupported detection model type: {model_type}")
    
    return det_model

# --------------------- CHOOSE OPTIMIZER AND SCHEDULER ---------------------- #

def create_optimizer_and_scheduler(optimizer_cfg, scheduler_cfg, model_params):
    """
    Creates an optimizer and scheduler based on the provided configuration.
    
    Args:
    - optimizer_cfg (dict): Configuration for the optimizer.
    - scheduler_cfg (dict): Configuration for the scheduler.
    - model_params (iterable): Parameters of the model to optimize.
    
    Returns:
    - optimizer (torch.optim.Optimizer): The configured optimizer.
    - scheduler (torch.optim.lr_scheduler): The configured scheduler.
    """
    
    # Create optimizer
    optimizer_type = optimizer_cfg.get('type', 'AdamW')  # Default to AdamW if not specified
    optimizer_params = {'lr': optimizer_cfg['lr'], 'weight_decay': optimizer_cfg['weight_decay']}

    if optimizer_type == 'AdamW':
        optimizer = AdamW(model_params, **optimizer_params)
    elif optimizer_type == 'SGD':
        optimizer = SGD(model_params, **optimizer_params)
    elif optimizer_type == 'RMSprop':
        optimizer = RMSprop(model_params, **optimizer_params)
    elif optimizer_type == 'Adam':
        optimizer = Adam(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Create scheduler
    scheduler_type = scheduler_cfg.get('type', 'CosineAnnealingWarmRestarts')  # Default to CosineAnnealingWarmRestarts
    if scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=scheduler_cfg['T_0'], T_mult=scheduler_cfg['T_mult'], eta_min=scheduler_cfg['eta_min'])
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_cfg['T_max'], eta_min=scheduler_cfg['eta_min'])
    elif scheduler_type == 'StepLR':
        scheduler = StepLR(optimizer, step_size=scheduler_cfg['step_size'], gamma=scheduler_cfg['gamma'])
    elif scheduler_type == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=scheduler_cfg['gamma'])
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return optimizer, scheduler
