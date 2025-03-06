
import os
import logging
import argparse
from monai.losses import DiceLoss
import torch
from torch import nn
import torch.distributed as dist

from src.utils import process_kwargs, load_experiment, summarize_config, get_project_root, determine_run_directory
from src.finetuning import (
    FinetuningTrainer,
    FinetuningDataModule,
    save_dataset_summary,
    create_optimizer_and_scheduler,
    load_segmentation_model,
    log_info,
    GPUSetup,
)
root = get_project_root()
# Retrieve a logger for the module
logger = logging.getLogger(__name__)

def datamodule(cfg, run_path=None):
    # -------------------- PREP DATASET INFO -------------------- #    
    dataset_cfg = dict()
    for dataset_name in cfg.get('dataset').keys():
        mask_labels = cfg.get('dataset').get(dataset_name).get('mask_labels')
        remove_label_ids = cfg.get('dataset').get(dataset_name).get('remove_label_ids')
        dataset_cfg[dataset_name] = {
            'ml_metadata_file': cfg.get('dataset').get(dataset_name).get('ml_metadata_file'),
            'slice_info_parquet_dir': cfg.get('dataset').get(dataset_name).get('slice_info_parquet_dir'),
            'mask_labels': mask_labels,
            'instance_bbox': cfg.get('dataset').get(dataset_name).get('instance_bbox'),
            'remove_label_ids': remove_label_ids
        }

    # Create an instance of FinetuningDataModule with your config parameters.
    finetune_dm = FinetuningDataModule(
        dataset_cfg=dataset_cfg,
        augmentation_config={
            'train': cfg.get('augmentation_pipeline', {}).get('train', None),
            'val': cfg.get('augmentation_pipeline', {}).get('val', None),
            'test': cfg.get('augmentation_pipeline', {}).get('test', None)
        },
        is_balanced=cfg.get('datamodule', {}).get('balanced', False),
        bbox_shift=cfg.get('datamodule', {}).get('bbox_shift', 0),
        max_subjects=cfg.get('datamodule', {}).get('max_subject_set', 'full')
    )
    
    # Retrieve DataLoaders and summaries from the data module.
    result = finetune_dm.get_loaders(
        batch_size=cfg.get('datamodule', {}).get('batch_size', 2),
        num_workers=cfg.get('datamodule', {}).get('num_workers', 1)
    )
    train_loader = result["loaders"]["train"]
    val_loader = result["loaders"]["val"]
    test_loader = result["loaders"]["test"]
    summaries = result["summaries"]
    
    # -------------------- SAVE DATASET SUMMARY -------------------- #
    log_info(f"max_train_subjects: {cfg.get('datamodule', {}).get('max_subject_set', 'full')}")
    if GPUSetup.is_main_process():
        summary_file_path = os.path.join(
            root,
            run_path,
            cfg.get('output_configuration').get('save_path'),
            cfg.get('output_configuration').get('summary_file')
        )
        save_dataset_summary(
            summaries,
            summary_file_path,
            max_subjects=cfg.get('datamodule', {}).get('max_subject_set', 'full')
        )
    
    # -------------------- RETURN DATALOADERS -------------------- #
    return train_loader, val_loader, test_loader

def prepare_training_base(trainable_cfg, model_config, weights_path, optimizer_cfg, scheduler_cfg,
                          checkpoint_path, device):
    log_info("Preparing training base on device: " + str(device))
    start_epoch = 0

    # Load segmentation model
    seg_model = load_segmentation_model(trainable_cfg, model_config, weights_path, device)

    print("Number of total parameters:", sum(p.numel() for p in seg_model.parameters()))
    print("Number of trainable parameters:", sum(p.numel() for p in seg_model.parameters() if p.requires_grad))

    # Resume from checkpoint if available
    if checkpoint_path and os.path.isfile(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            seg_model.load_state_dict(checkpoint["model"])
            log_info("Resuming training from epoch " + str(start_epoch) +
                     " with checkpoint " + checkpoint_path)
        except Exception as e:
            logger.error("Failed to load checkpoint from " + checkpoint_path + ": " + str(e))
            raise e

    # Get parameters that require gradient updates
    trainable_params = [param for param in seg_model.parameters() if param.requires_grad]

    # Create optimizer and scheduler using a utility function
    optimizer, scheduler = create_optimizer_and_scheduler(optimizer_cfg, scheduler_cfg, trainable_params)

    # Define loss functions
    loss_fn = (DiceLoss(sigmoid=True, squared_pred=True, reduction="mean"),
               nn.BCEWithLogitsLoss(reduction="mean"))

    # Resume optimizer state if checkpoint exists
    if checkpoint_path and os.path.isfile(checkpoint_path):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return seg_model, optimizer, scheduler, loss_fn, start_epoch

def finetune_main_impl(cfg):
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
    module_cfg = cfg.get('module', {})
    model, optimizer, scheduler, loss_fn, start_epoch = prepare_training_base(
        trainable_cfg=module_cfg.get('trainable', {}),
        model_config=module_cfg.get('model_type', 'vit_b'),
        weights_path=module_cfg.get('pretrain_model'),
        optimizer_cfg=module_cfg.get('optimizer'),
        scheduler_cfg=module_cfg.get('scheduler'),
        checkpoint_path=module_cfg.get('checkpoint'),
        device=device
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

    summarize_config(cfg, path=os.path.join(root, run_path, cfg.get('output_configuration').get('save_path')))
    
    # --------------- TRAIN --------------- #
    trainer = FinetuningTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler, 
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        module_cfg=module_cfg,
        datamodule_cfg=cfg.get('datamodule', {}),
        experiment_cfg=cfg.get('experiment', {}),
        run_path=run_path,
        device=device,
        start_epoch=start_epoch,
    )
    # Start training and validation phases
    # logger.info(f"Local Rank {local_rank}: Starting training and validation phases...")
    trainer.train(num_epochs=cfg['module']['num_epochs'])

def finetune_main(config, kwargs=None):
    """
    Accepts either a configuration file name (string) or a loaded configuration (dict).
    If a string is provided, the configuration is loaded using any additional keyword arguments.
    """
    import traceback
    if kwargs is None:
        kwargs = {}
    if isinstance(config, str):
        config_file = config if config.endswith('.yaml') else config + '.yaml'
        cfg = load_experiment(config_file, root, kwargs)
    else:
        cfg = config

    logger = GPUSetup.setup_logging(
        config_level=cfg.get('output_configuration', {}).get('logging_level', 'INFO').upper(),
        logger=logging.getLogger(__name__)
    )
    GPUSetup.setup(distributed=cfg.get('distributed', False),
                   seed=cfg.get('SEED', 42))
    try:
        finetune_main_impl(cfg)
    except Exception:
        logger.exception("An error occurred")
    finally:
        GPUSetup.cleanup()
        log_info("Cleanup completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning model setup.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    parser.add_argument('--kwargs', nargs=argparse.REMAINDER, help="Additional command line key=value pairs")
    args = parser.parse_args()
    extra_kwargs = process_kwargs(args.kwargs)
    finetune_main(args.config_name, extra_kwargs)
