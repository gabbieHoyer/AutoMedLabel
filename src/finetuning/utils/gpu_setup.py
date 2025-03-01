import os
import random
import logging
import numpy as np

import torch
import torch.distributed as dist

# -------------------- GPU Set Up --------------------
def setup(distributed:bool=False, seed:int=42):
    """
    Setup for multi-GPU
    """
    def set_environment_variables():
        """
        Ensures that the threading environment variables are appropriately set to optimize performance on single GPU.
        """
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["OPENBLAS_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "6"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
        os.environ["NUMEXPR_NUM_THREADS"] = "6"

    # distributed training setup
    def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    # Initialize defaults
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if distributed:
        # Set the MASTER_ADDR and MASTER_PORT from the environment or use defaults
        master_addr = os.getenv('MASTER_ADDR', 'localhost')

        print(os.getenv('MASTER_PORT'), flush=True)

        master_port = os.getenv('MASTER_PORT', '5675')

        print(master_port, flush=True)

        dist_backend = os.getenv('DIST_BACKEND', 'nccl')
        dist_url = os.getenv('DIST_URL', f"tcp://{master_addr}:{master_port}")
        
        # Initialize the distributed environment
        dist.init_process_group(
            backend=dist_backend,
            init_method=dist_url,
            rank=rank,
            world_size=world_size
        )
        dist.barrier()
        setup_for_distributed(rank == 0)
        print(f'World size: {world_size}; Rank: {rank}; LocalRank: {local_rank}', flush=True)

    else:
        # For non-distributed training, default to local_rank 0
        set_environment_variables()
    
    # Set the device based on local_rank
    device_ = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_)
    if 'cuda' in device_:
        torch.cuda.set_device(device)

    # Set the seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Configure PyTorch's cudnn for the system's characteristics
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_distributed():
    return dist.is_initialized()

def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1

def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def cleanup():
    if is_distributed():
        dist.destroy_process_group()
    torch.cuda.empty_cache()  # Helps in releasing unreferenced memory immediately
    # You can add any other cleanup operations here

# ----------------- LOGGING -----------------

def setup_logging(config_level, logger):
    # Mapping configuration levels to logging module levels
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    # Map config_level to a logging module level, defaulting to INFO if not found
    logging_level = level_mapping.get(config_level, logging.INFO)
    
    rank = get_rank()  # Retrieve the current process rank once
    logging.basicConfig(
        level=logging_level,
        format=f'%(asctime)s - %(levelname)s - Rank {rank} - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Number of GPUs available: %s", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info("GPU %s: %s", i, torch.cuda.get_device_name(i))
        logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'))

    return logger

