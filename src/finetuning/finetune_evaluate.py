
import os
import copy
import logging
import argparse

import torch
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
from src.finetuning.utils.logging import log_info
from src.finetuning.engine.models.sam import finetunedSAM
from src.finetuning.utils.utils import determine_run_directory
from src.finetuning.engine.finetune_engine_eval import Tester 
from src.utils.file_management.config_handler import load_evaluation, summarize_config
from src.finetuning.datamodule.eval_dataset_loader import get_dataset_info, process_dataset, create_dataloader
 
# Retrieve a logger for the module
logger = logging.getLogger(__name__)

def datamodule(cfg, run_path=None):
    # -------------------- PREP DATASET INFO -------------------- #    
    dataset_cfg = dict()
    for dataset_name in cfg.get('dataset').keys():
        mask_labels = cfg.get('dataset').get(dataset_name).get('mask_labels')
        remove_label_ids = cfg.get('dataset').get(dataset_name).get('remove_label_ids')
        instance_bbox = cfg.get('dataset').get(dataset_name).get('instance_bbox')

        dataset_cfg[dataset_name] = {'ml_metadata_file': cfg.get('dataset').get(dataset_name).get('ml_metadata_file'),
                                     'slice_info_parquet_dir': cfg.get('dataset').get(dataset_name).get('slice_info_parquet_dir'),
                                     'mask_labels': mask_labels,
                                     'instance_bbox': instance_bbox,
                                     'remove_label_ids': remove_label_ids
                                     }
    
    dataset_info = get_dataset_info(dataset_cfg = dataset_cfg, 
                                 is_balanced = cfg.get('datamodule', {}).get('balanced', False)
                                 )
    
    cfg['datamodule']['dataset_name'] = dataset_name
    cfg['datamodule']['num_classes'] = len(mask_labels)
    cfg['datamodule']['mask_labels'] = mask_labels
    cfg['datamodule']['remove_label_ids'] = remove_label_ids
    cfg['datamodule']['instance_bbox'] = instance_bbox

    # -------------------- EXTRACT DATASET INFO -------------------- #
    sampled_test_dataset, full_test_dataset = process_dataset(
            dataset_info = dataset_info,
            bbox_shift = cfg.get("bbox_shift", 0),
            max_subjects = cfg.get('datamodule', {}).get('max_subject_set', 'full')
            )

    # -------------------- EXTRACT DATALOADERS -------------------- # 
    # NOTE: Eval only verified for batch_size = 1
       
    sampled_test_loader, full_test_loader = create_dataloader(
            datasets = (sampled_test_dataset, full_test_dataset), 
            batch_size = 1,
            num_workers = cfg.get('datamodule', {}).get('num_workers', 1),
            instance_bbox=instance_bbox
            )
    
    return sampled_test_loader, full_test_loader


def prepare_training_base(comp_cfg, model_type:str, initial_weights:str, finetuned_weights:str, device):
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

    # Check if finetuned_weights are the same as initial_weights
    if finetuned_weights == initial_weights:
        log_info("finetuned_weights and initial_weights are the same. Skipping checkpoint loading.")
        return finetuned_model

    # Check if a checkpoint exists to resume training
    if finetuned_weights and os.path.isfile(finetuned_weights):
        try:
            checkpoint = torch.load(finetuned_weights, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            finetuned_model.load_state_dict(checkpoint["model"])
            log_info(f"Resuming training from epoch {start_epoch} with checkpoint {finetuned_weights}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {finetuned_weights}: {e}")
            # Decide whether to continue with training from scratch or to abort
            raise e

    return finetuned_model


def finetune_evaluate(cfg):
    # --------------- SET UP ENVIRONMENT --------------- #  
    rank = GPUSetup.get_rank()
    ngpus_per_node = torch.cuda.device_count()

    # Detect if we have a GPU available and choose device accordingly
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    logger.info(f"Local Rank {local_rank}: Starting finetune_evaluate")

    # ------------- SET UP EXPERIMENT RUN  ------------- #
    module_cfg = cfg.get('module')
    model_cfg = cfg.get('model')

    for key, val in cfg['dataset'].items():
        task_name = cfg['dataset'][key]['project']

    module_cfg['task_name'] = task_name

    group_name = f"{model_cfg['model_details']['subject']}_trainSubjects_bboxShift_{model_cfg['model_details']['bbox_shift']}_sliceBalance_{model_cfg['model_details']['balanced']}_imgEnc_{model_cfg['trainable']['image_encoder']}_maskDec_{model_cfg['trainable']['mask_decoder']}"

    if GPUSetup.is_distributed():
        # If distributed training is enabled, synchronize creation of the run directory
        if GPUSetup.is_main_process(): 
            # Only main process determines the run directory
            run_path = determine_run_directory(module_cfg['work_dir'], module_cfg['task_name'], os.path.join(model_cfg['model_weights'].split('-', 1)[0], group_name))

            # Since run_path is a string, use broadcast_object_list
            dist.broadcast_object_list([run_path], src=0)  # src=0 denotes the main process
        else:
            # Receive broadcasted run_path
            run_path = [None]  # Placeholder for the received object
            dist.broadcast_object_list(run_path, src=0)
            run_path = run_path[0]  # Unpack the list to get the actual path
    else:
        # If not distributed, directly determine the run directory
        run_path = determine_run_directory(module_cfg['work_dir'], module_cfg['task_name'], os.path.join(model_cfg['model_weights'].split('-', 1)[0], group_name))

    log_info(f"Run path: {run_path}")

    # --------------- SET UP DATALOADERS --------------- #
    sampled_test_loader, full_test_loader = datamodule(cfg, run_path)

    # --------------- SET UP MODEL --------------- #
    # Initialize model, optimizer, loss functions, and potentially load checkpoint
    model = prepare_training_base(
            comp_cfg = model_cfg.get('trainable', {}),
            model_type = model_cfg.get('model_type'), 
            initial_weights = cfg.get('base_model'),
            finetuned_weights = model_cfg.get('finetuned_model'),
            device = device, 
            )
    
    if GPUSetup.is_distributed():
        # Convert all BatchNorm layers to SyncBatchNorm layers
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], \
            broadcast_buffers=True, find_unused_parameters=True)

        torch.backends.cudnn.benchmark = True

    # --------------- EVALUATE --------------- #
    summarize_config(cfg, path=os.path.join(root, run_path, cfg.get('output_configuration').get('save_path')))
    
    # Instantiate the Tester and run tests depending on available data loaders

    if sampled_test_loader is not None:
        # Testing with sampled data
        sampled_tester = Tester(
            model=model,
            test_loader=sampled_test_loader,
            eval_cfg=model_cfg,
            module_cfg=cfg.get('module'),
            datamodule_cfg=cfg.get('datamodule'),
            experiment_cfg=cfg.get('evaluation'),
            run_path=run_path,
            device=device,
            data_type='sampled',
            visualize=cfg.get('module').get('visualize')
        )
        logger.info(f"Local Rank {local_rank}: Starting testing phase with sampled data...")
        sampled_tester.test()

    # Testing with full data
    full_tester = Tester(
        model=model,
        test_loader=full_test_loader,
        eval_cfg=model_cfg,
        module_cfg=cfg.get('module'),
        datamodule_cfg=cfg.get('datamodule'),
        experiment_cfg=cfg.get('evaluation'),
        run_path=run_path,
        device=device,
        data_type='full',
        visualize=cfg.get('module').get('visualize')
    )

    logger.info(f"Local Rank {local_rank}: Starting testing phase with full data...")
    full_tester.test()

    # Clean up wandb (if needed)
    if module_cfg.get('use_wandb', False) and GPUSetup.is_main_process():
        import wandb
        wandb.finish()
        log_info("wandb finish.")


if __name__ == "__main__":
    try:
        # Argument parsing to allow for flexible execution of the script
        parser = argparse.ArgumentParser(description="Finetuning Evaluation setup.")
        parser.add_argument("config_name", help="Name of the YAML configuration file")
        args = parser.parse_args()

        # Construct the configuration file name and load the experiment configuration
        config_name = args.config_name + '.yaml'  # Append '.yaml' to the provided config name

        cfg = load_evaluation(config_name, root)  

        for model_info in cfg.get('models', []):
            # Create a copy of the main config and update it with the current model's information
            model_config = copy.deepcopy(cfg)

            model_config['model'] = model_info

            # Setup logging for the application
            # Get the logging level from the configuration, default to 'INFO' if not found
            logger = GPUSetup.setup_logging(
                    config_level = cfg.get('output_configuration', {}).get('logging_level', 'INFO').upper(),
                    logger = logging.getLogger(__name__))

            # Setup environment for distributed training, if applicable
            GPUSetup.setup(distributed = cfg.get('distributed', False),
                        seed = cfg.get('SEED', 42))

            finetune_evaluate(model_config)

            # dont think this is necessary since I'm doing it in the func above:
            # # Clean up wandb (if needed)
            # if model_config.get('use_wandb', False) and GPUSetup.is_main_process():
            #     import wandb
            #     wandb.finish()
            #     log_info("wandb finish.")
            
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





