""" Finetuning scripts functional for non instance-based finetuning """

import os
import logging
import argparse
import yaml
import monai
from monai.losses import DiceLoss

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from segment_anything import sam_model_registry

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import src.finetuning.utils.gpu_setup as GPUSetup
from src.finetuning.utils.utils import determine_run_directory
from src.utils.file_management.config_handler import load_experiment, summarize_config
from src.utils.file_management.args_handler import process_kwargs, apply_overrides
from src.finetuning.datamodule.dynamic_dataset_loader import get_dataset_info, process_dataset, save_dataset_summary, create_dataloader
from src.finetuning.engine.finetune_engine import Trainer, Tester  
from src.finetuning.engine.models.sam import finetunedSAM
from src.finetuning.utils.logging import log_info

# Retrieve a logger for the module
logger = logging.getLogger(__name__)

def datamodule(cfg, run_path=None):
    # -------------------- PREP DATASET INFO -------------------- #    
    
    dataset_cfg = dict()
    for dataset_name in cfg.get('dataset').keys():
        mask_labels = cfg.get('dataset').get(dataset_name).get('mask_labels')
        remove_label_ids = cfg.get('dataset').get(dataset_name).get('remove_label_ids')

        dataset_cfg[dataset_name] = {'ml_metadata_file': cfg.get('dataset').get(dataset_name).get('ml_metadata_file'),
                                     'slice_info_parquet_dir': cfg.get('dataset').get(dataset_name).get('slice_info_parquet_dir'),
                                     'mask_labels': mask_labels,
                                     'instance_bbox': cfg.get('dataset').get(dataset_name).get('instance_bbox'),
                                     'remove_label_ids': remove_label_ids
                                    }

    dataset_info = get_dataset_info(dataset_cfg = dataset_cfg, 
                            is_balanced = cfg.get('datamodule', {}).get('balanced', False)
                            )

    # -------------------- EXTRACT DATASET INFO -------------------- #
    train_dataset, val_dataset, test_dataset, summaries = process_dataset(
            dataset_info = dataset_info,
            augmentation_config = {'train': cfg.get('augmentation_pipeline', {}).get('train', None),
                                   'val':  cfg.get('augmentation_pipeline', {}).get('val', None),
                                   'test': cfg.get('augmentation_pipeline', {}).get('test', None)},
            bbox_shift = cfg.get('datamodule', {}).get("bbox_shift", 0),
            max_subjects = cfg.get('datamodule', {}).get('max_subject_set', 'full')
            )

            # bbox shift change after t1sag spine is done

    # -------------------- SAVE DATASET SUMMARY -------------------- #   
    log_info(f"max_train_sujects: {cfg.get('datamodule', {}).get('max_subject_set', 'full')}")

    if GPUSetup.is_main_process():
        summary_file_path = os.path.join(root, run_path, cfg.get('output_configuration').get('save_path'), cfg.get('output_configuration').get('summary_file'))
        save_dataset_summary(summaries, summary_file_path, max_subjects = cfg.get('datamodule', {}).get('max_subject_set', 'full'))
    
    # -------------------- EXTRACT DATALOADERS -------------------- #    
    train_loader, val_loader, test_loader = create_dataloader(
            datasets = (train_dataset, val_dataset, test_dataset), 
            batch_size = cfg.get('datamodule', {}).get('batch_size', 2),
            num_workers = cfg.get('datamodule', {}).get('num_workers', 1),
            )
    
    return train_loader, val_loader, test_loader


def prepare_training_base(comp_cfg, model_type:str, initial_weights:str, optimizer_cfg:dict, scheduler_cfg:dict, resume_checkpoint_path:str, device):
    log_info(f"Preparing training base on {device}")

    start_epoch = 0  # Default to starting from scratch

    sam_model = sam_model_registry[model_type](checkpoint=initial_weights)

    # Model finetuning setup
    finetuned_model = finetunedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        config=comp_cfg
    ).to(device)

    # Check if a checkpoint exists to resume training
    if resume_checkpoint_path and os.path.isfile(resume_checkpoint_path):
        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            finetuned_model.load_state_dict(checkpoint["model"])
            log_info(f"Resuming training from epoch {start_epoch} with checkpoint {resume_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {resume_checkpoint_path}: {e}")
            # Decide whether to continue with training from scratch or to abort
            raise e

    # Prepare optimizer with parameters that require gradients
    img_mask_encdec_params = [param for param in finetuned_model.parameters() if param.requires_grad]
    
    optimizer = AdamW(img_mask_encdec_params, lr=optimizer_cfg['lr'], weight_decay=optimizer_cfg['weight_decay'])

    # scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_cfg['T_max'], eta_min=scheduler_cfg['eta_min'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=scheduler_cfg['eta_min'])

    # Prepare loss functions
    loss_fn = (DiceLoss(sigmoid=True, squared_pred=True, reduction="mean"), nn.BCEWithLogitsLoss(reduction="mean"))

    # Load optimizer state if resuming from checkpoint
    if resume_checkpoint_path and os.path.isfile(resume_checkpoint_path):
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    return finetuned_model, optimizer, scheduler, loss_fn, start_epoch


def finetune_main(cfg):
    # --------------- SET UP ENVIRONMENT --------------- #  
    rank = GPUSetup.get_rank()
    ngpus_per_node = torch.cuda.device_count()

    # Detect if we have a GPU available and choose device accordingly
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    logger.info(f"Local Rank {local_rank}: Starting finetune_main")

    if GPUSetup.is_distributed():
        if rank % ngpus_per_node == 0:
            print("Before DDP initialization:", flush=True)
            os.system("nvidia-smi")

    # ------------- SET UP EXPERIMENT RUN  ------------- #
    module_cfg = cfg.get('module')

    if GPUSetup.is_distributed():
        # If distributed training is enabled, synchronize creation of the run directory
        group_name = f"{cfg.get('experiment').get('pretrained_weights')}_{cfg.get('datamodule').get('max_subject_set')}_trainSubjects_sliceBalance_{cfg.get('datamodule').get('balanced')}_imgEnc_{module_cfg.get('trainable').get('image_encoder')}_maskDec_{module_cfg.get('trainable').get('mask_decoder')}"
        
        if GPUSetup.is_main_process(): 
            # Only main process determines the run directory
            run_path = determine_run_directory(module_cfg['work_dir'], module_cfg['task_name'], group_name)
            # Since run_path is a string, use broadcast_object_list
            dist.broadcast_object_list([run_path], src=0)  # src=0 denotes the main process
        else:
            # Receive broadcasted run_path
            run_path = [None]  # Placeholder for the received object
            dist.broadcast_object_list(run_path, src=0)
            run_path = run_path[0]  # Unpack the list to get the actual path
    else:
        # If not distributed, directly determine the run directory
        run_path = determine_run_directory(module_cfg['work_dir'], module_cfg['task_name'], group_name)

    # --------------- SET UP DATALOADERS --------------- #
    train_loader, val_loader, test_loader = datamodule(cfg, run_path)

    # --------------- SET UP MODEL --------------- #
    # Initialize model, optimizer, loss functions, and potentially load checkpoint
    model, optimizer, scheduler, loss_fn, start_epoch = prepare_training_base(
            comp_cfg = cfg.get('module').get('trainable', {}),
            model_type = cfg.get('module').get('model_type'), 
            initial_weights = cfg.get('module').get('pretrain_model'),
            optimizer_cfg = {'lr': cfg.get('module').get('optimizer').get('lr'), 
                        'weight_decay': cfg.get('module').get('optimizer').get('weight_decay')},
            scheduler_cfg = {'eta_min': cfg.get('module').get('scheduler').get('eta_min')},
            resume_checkpoint_path = cfg.get('module').get('checkpoint'),
            device = device, 
            )
    
    if GPUSetup.is_distributed():
        # Convert all BatchNorm layers to SyncBatchNorm layers
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], \
            broadcast_buffers=True, find_unused_parameters=True)

        torch.backends.cudnn.benchmark = True

    if GPUSetup.is_distributed():
        if rank % ngpus_per_node == 0:
            print("After DDP initialization:", flush=True)
            os.system("nvidia-smi")

    # --------------- TRAIN --------------- #
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler, 
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        module_cfg=cfg.get('module'),
        datamodule_cfg=cfg.get('datamodule'),
        experiment_cfg=cfg.get('experiment'),
        run_path=run_path,
        device=device,
        start_epoch=start_epoch,
    )

    summarize_config(cfg, path=os.path.join(root, run_path, cfg.get('output_configuration').get('save_path')))
    
    # Start training and validation phases
    logger.info(f"Local Rank {local_rank}: Starting training and validation phases...")
    trainer.train(num_epochs=cfg['module']['num_epochs'])

    # --------------- EVALUATE --------------- #
    # # Instantiate the Tester with the appropriate loaders and model setup
    # tester = Tester(
    #     model=model,
    #     test_loader=test_loader,
    #     module_cfg=cfg.get('module'),
    #     experiment_cfg=cfg.get('experiment'),
    #     run_path=run_path,
    #     device=device,
    # )
    # logger.info(f"Local Rank {local_rank}: Starting testing phase...")
    # tester.test()



if __name__ == "__main__":
    try:
        # Argument parsing to allow for flexible execution of the script
        parser = argparse.ArgumentParser(description="Fine-tuning model setup.")
        parser.add_argument("config_name", help="Name of the YAML configuration file")
        parser.add_argument('--kwargs', nargs=argparse.REMAINDER, help="Additional command line key=value pairs")
        args = parser.parse_args()
        kwargs = process_kwargs(args.kwargs)

        # Construct the configuration file name and load the experiment configuration
        config_name = args.config_name + '.yaml'  

        cfg = load_experiment(config_name, root, kwargs) 

        # Setup logging for the application
        logger = GPUSetup.setup_logging(
                config_level = cfg.get('output_configuration', {}).get('logging_level', 'INFO').upper(),
                logger = logging.getLogger(__name__))

        # Setup environment for distributed training, if applicable
        GPUSetup.setup(distributed = cfg.get('distributed', False),
                       seed = cfg.get('SEED', 42))

        finetune_main(cfg)
            
    except Exception as e:
        if logger is not None:
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")
            # handle any specific exceptions or perform any actions before cleanup
    finally:
        if 'GPUSetup' in locals():
            GPUSetup.cleanup()
            log_info("Cleanup completed.")






