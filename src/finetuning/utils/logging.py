import logging
from functools import wraps

from . import gpu_setup as GPUSetup

logger = logging.getLogger(__name__)

# -------- DECORATOR FOR MAIN PROCESS ONLY FUNCTIONALITY -------- #
def main_process_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if GPUSetup.is_main_process():
            return func(*args, **kwargs)
    return wrapper

@main_process_only
def log_info(message):
    logger.info(message, stacklevel=2)

@main_process_only
def wandb_log(data):
    import wandb  
    wandb.log(data)