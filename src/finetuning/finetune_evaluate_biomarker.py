import os
import copy
import logging
import argparse
import torch
import torch.distributed as dist

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)
from src.utils import determine_run_directory, load_evaluation, summarize_config
from src.finetuning import GPUSetup, log_info, CustomMetricTester, BiomarkerEvaluationDataModule, load_segmentation_model

# Retrieve a logger for the module
logger = logging.getLogger(__name__)

def datamodule(cfg, run_path=None):
    # -------------------- PREP DATASET INFO -------------------- #    
    dataset_cfg = dict()
    for dataset_name in cfg.get('dataset').keys():
        mask_labels = cfg.get('dataset').get(dataset_name).get('mask_labels')
        remove_label_ids = cfg.get('dataset').get(dataset_name).get('remove_label_ids')
        instance_bbox = cfg.get('dataset').get(dataset_name).get('instance_bbox')
        stats_metadata_file = cfg.get('dataset').get(dataset_name).get('stats_metadata_file')

        dataset_cfg[dataset_name] = {
            'ml_metadata_file': cfg.get('dataset').get(dataset_name).get('ml_metadata_file'),
            'slice_info_parquet_dir': cfg.get('dataset').get(dataset_name).get('slice_info_parquet_dir'),
            'mask_labels': mask_labels,
            'instance_bbox': instance_bbox,
            'remove_label_ids': remove_label_ids
        }

    cfg['datamodule']['dataset_name'] = dataset_name
    cfg['datamodule']['num_classes'] = len(mask_labels)
    cfg['datamodule']['mask_labels'] = mask_labels
    cfg['datamodule']['remove_label_ids'] = remove_label_ids
    cfg['datamodule']['instance_bbox'] = instance_bbox
    cfg['datamodule']['stats_metadata_file'] = stats_metadata_file

    # Instantiate the EvaluationDataModule.
    eval_dm = BiomarkerEvaluationDataModule(
        dataset_cfg=dataset_cfg,
        is_balanced=cfg.get('datamodule', {}).get('balanced', False),
        bbox_shift=cfg.get("bbox_shift", 0),
        max_subjects=cfg.get('datamodule', {}).get('max_subject_set', 'full'),
        metric_type=cfg.get('datamodule', {}).get('metric').get('func')
    )
    loaders_dict = eval_dm.get_loaders(batch_size=1, num_workers=cfg.get('datamodule', {}).get('num_workers', 1), instance_bbox=instance_bbox)

    return loaders_dict["loaders"]["sampled"], loaders_dict["loaders"]["full"]

def prepare_training_base(trainable_cfg, model_config, weights_path, finetuned_weights, device):
    log_info("Preparing training base on device: " + str(device))
    start_epoch = 0

    # Load segmentation model
    seg_model = load_segmentation_model(trainable_cfg, model_config, weights_path, device)
    
    # If no finetuned model is provided, continue using the pretrained model weights.
    if not finetuned_weights or finetuned_weights.strip() == "":
        log_info("No finetuned model provided. Using base model weights.")
        return seg_model

    # Resume from checkpoint if available
    if finetuned_weights and os.path.isfile(finetuned_weights):
        try:
            checkpoint = torch.load(finetuned_weights, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            seg_model.load_state_dict(checkpoint["model"])
            log_info("Evaluating model, previously finetuned to epoch " + str(start_epoch) +
                     " with checkpoint " + finetuned_weights)
        except Exception as e:
            logger.error("Failed to load checkpoint from " + finetuned_weights + ": " + str(e))
            raise e

    # Get parameters that require gradient updates
    trainable_params = [param for param in seg_model.parameters() if param.requires_grad]

    return seg_model

def finetune_evaluate_biomarker_impl(cfg):
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
            run_path = determine_run_directory(module_cfg['work_dir'], module_cfg['task_name'], os.path.join(model_cfg['model_identifier'].split('-', 1)[0], group_name))

            # Since run_path is a string, use broadcast_object_list
            dist.broadcast_object_list([run_path], src=0)  # src=0 denotes the main process
        else:
            # Receive broadcasted run_path
            run_path = [None]  # Placeholder for the received object
            dist.broadcast_object_list(run_path, src=0)
            run_path = run_path[0]  # Unpack the list to get the actual path
    else:
        # If not distributed, directly determine the run directory
        run_path = determine_run_directory(module_cfg['work_dir'], module_cfg['task_name'], os.path.join(model_cfg['model_identifier'].split('-', 1)[0], group_name))

    log_info(f"Run path: {run_path}")

    # --------------- SET UP DATALOADERS --------------- #
    sampled_test_loader, full_test_loader = datamodule(cfg, run_path)

    # --------------- SET UP MODEL --------------- #
    # Initialize model, optimizer, loss functions, and potentially load checkpoint
    model = prepare_training_base(
        trainable_cfg=model_cfg.get('trainable', {}),
        model_config=model_cfg.get('model_type', 'vit_b'),
        weights_path=model_cfg.get('pretrain_model'),
        finetuned_weights=model_cfg.get('finetuned_model'),
        device=device
    )
       
    if GPUSetup.is_distributed():
        # Convert all BatchNorm layers to SyncBatchNorm layers
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], \
            broadcast_buffers=True, find_unused_parameters=True)

        torch.backends.cudnn.benchmark = True

    # --------------- EVALUATE --------------- #
    summarize_config(cfg, path=os.path.join(root, run_path, cfg.get('output_configuration').get('save_path')))

    print(f'length of sampled data: {len(sampled_test_loader)}')
    # Instantiate the Tester and run tests depending on available data loaders
    if sampled_test_loader is not None:
        # Testing with sampled data
        sampled_tester = CustomMetricTester(
            model=model,
            test_loader=sampled_test_loader,
            eval_cfg=model_cfg,
            module_cfg=module_cfg,
            datamodule_cfg=cfg.get('datamodule'),
            experiment_cfg=cfg.get('evaluation'),
            run_path=run_path,
            device=device,
            data_type='sampled',
            visualize=module_cfg.get('visualize')
        )
        logger.info(f"Local Rank {local_rank}: Starting testing phase with sampled data...")
        sampled_tester.test()

    print(f'length of full data: {len(full_test_loader)}')
    # Testing with full data
    full_tester = CustomMetricTester(
        model=model,
        test_loader=full_test_loader,
        eval_cfg=model_cfg,
        module_cfg=module_cfg,
        datamodule_cfg=cfg.get('datamodule'),
        experiment_cfg=cfg.get('evaluation'),
        run_path=run_path,
        device=device,
        data_type='full',
        visualize=module_cfg.get('visualize')
    )
    logger.info(f"Local Rank {local_rank}: Starting testing phase with full data...")
    full_tester.test()

    # Clean up wandb (if needed)
    if module_cfg.get('use_wandb', False) and GPUSetup.is_main_process():
        import wandb
        wandb.finish()
        log_info("wandb finish.")

def finetune_evaluate_biomarker(config):
    """
    Accepts either a configuration file name (string) or a loaded configuration (dict).
    If a string is provided the configuration is loaded and each model is processed.
    """
    import traceback
    if isinstance(config, str):
        config_file = config if config.endswith('.yaml') else config + '.yaml'
        cfg = load_evaluation(config_file, root)
    else:
        cfg = config

    for model_info in cfg.get('models', []):
        model_config = copy.deepcopy(cfg)
        model_config['model'] = model_info

        logger = GPUSetup.setup_logging(
            config_level=cfg.get('output_configuration', {}).get('logging_level', 'INFO').upper(),
            logger=logging.getLogger(__name__)
        )
        GPUSetup.setup(distributed=cfg.get('distributed', False),
                       seed=cfg.get('SEED', 42))
        try:
            finetune_evaluate_biomarker_impl(model_config)
        except Exception:
            logger.exception("An error occurred")
        finally:
            GPUSetup.cleanup()
            logger.info("Cleanup completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Biomarker Evaluation setup.")
    parser.add_argument("config_name", help="Name of the YAML configuration file")
    args = parser.parse_args()
    finetune_evaluate_biomarker(args.config_name)

