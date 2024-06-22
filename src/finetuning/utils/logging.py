from functools import wraps
import logging

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import src.finetuning.utils.gpu_setup as GPUSetup #is_distributed

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
    logger.info(message)

@main_process_only
def wandb_log(data):
    import wandb  # It's a good practice to import within the function if it's not used elsewhere to avoid unnecessary imports
    wandb.log(data)