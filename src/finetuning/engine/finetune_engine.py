""" Finetuning scripts functional for non instance-based finetuning """

import logging
from datetime import datetime
from monai.metrics import DiceMetric

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import src.finetuning.utils.gpu_setup as GPUSetup #is_distributed
from src.finetuning.utils.utils import plot_combined_losses, plot_losses, save_losses, reduce_tensor, plot_metrics, save_metrics
from src.finetuning.utils.checkpointing import log_and_checkpoint, final_checkpoint_conversion
from src.finetuning.utils.logging import main_process_only, log_info, wandb_log
from src.finetuning.utils.fig_QC import visualize_input

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn, train_loader, val_loader, module_cfg, datamodule_cfg, experiment_cfg, run_path, device='cpu', start_epoch=0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.module_cfg = module_cfg
        self.datamodule_cfg = datamodule_cfg
        self.experiment_cfg = experiment_cfg
        self.run_path = run_path  

        self.device = device
        self.use_amp = module_cfg.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.current_step = 0  # Initialize the step counter here

        self.model_save_path, self.run_id = self.setup_experiment_environment()
        self.start_epoch = start_epoch

        # Early stopping configuration
        self.early_stopping_enabled = module_cfg.get('early_stopping', {}).get('enabled', True)
        self.patience = module_cfg.get('early_stopping', {}).get('patience', 5)
        self.min_delta = module_cfg.get('early_stopping', {}).get('min_delta', 0.001)
        self.early_stopped = False

        # Initialize DiceMetric for both training and validation
        self.shared_losses = {'train': [], 'val': []}
        self.shared_metrics = {'train': [], 'val': []}
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    def setup_experiment_environment(self):
        # Use the pre-determined run_path
        model_save_path = self.run_path
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        
        # Initialize Weights & Biases if configured
        if self.module_cfg.get('use_wandb', False) and GPUSetup.is_main_process():
            import wandb
            wandb.login()
            wandb.init(project=self.module_cfg['task_name'], 
                    group=self.module_cfg['group_name'], 
                    config={
                        "max_subject_set": self.datamodule_cfg['max_subject_set'],
                        "balanced": self.datamodule_cfg['balanced'],
                        "lr": self.module_cfg['optimizer']['lr'],
                        "batch_size": self.datamodule_cfg['batch_size'],
                        "grad_accum": self.module_cfg['grad_accum'],
                        "model_type": self.module_cfg['sam2_model_cfg'],  #self.module_cfg['model_type'],
                        "description": self.experiment_cfg['description'],
                        "pretrained_weights": self.experiment_cfg['pretrained_weights'],
                        "image_encoder": self.module_cfg['trainable']['image_encoder'],
                        "mask_decoder": self.module_cfg['trainable']['mask_decoder'],
                        "early_stopping": self.module_cfg['early_stopping']['enabled'],
                        "patience": self.module_cfg['early_stopping']['patience'],
                        "bbox_shift": self.datamodule_cfg['bbox_shift'],
                    }, 
                    settings=wandb.Settings(_service_wait=300),
                    tags=['experiment', self.experiment_cfg['name']],
                    name=run_id)  # Use run_id as the name for the W&B run for easy identification

        return model_save_path, run_id

    def calculate_loss(self, predictions, gt2D, seg_loss_weight=0.5, ce_loss_weight=0.5):
        seg_loss, ce_loss = self.loss_fn
        total_loss = seg_loss_weight * seg_loss(predictions, gt2D) + ce_loss_weight * ce_loss(predictions, gt2D.float())
        return total_loss

    def process_batch(self, batch, mode='train', batch_idx=0, epoch=0):
        # Non-tensor data, keep on CPU
        dataset_name, img_name = batch['dataset_name'], batch['img_name'] 

        # Transfer all tensor data in the batch to the specified device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)

        # Now, extract the tensors directly from the batch
        image = batch['image']   # torch.Size([2, 3, 1024, 1024])
        gt2D = batch['gt2D']     # torch.Size([2, 1, 1024, 1024])
        boxes = batch['boxes']   # torch.Size([2, 4])
        label_id = batch['label_id']

        if mode == 'train':
            accumulate_steps = self.module_cfg.get('grad_accum', 1)  # Get accumulation steps from config, default is 1
            if self.current_step % accumulate_steps == 0:  # Only zero gradients on the correct accumulation step
                self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):  # Conditional AMP
                predictions = self.model(image, boxes)

                loss = self.calculate_loss(predictions, gt2D) / accumulate_steps  # Normalize the loss

            if self.module_cfg['visualize'] and epoch == 0 and batch_idx < 1 and GPUSetup.is_main_process():  # Visualize only for the first 5 batches
                with torch.no_grad():
                    
                    # Detach and clone necessary tensors for visualization to minimize impact on training
                    image_for_viz = image.detach().cpu()
                    gt_mask_for_viz = gt2D.detach().cpu()
                    pred_mask_for_viz = predictions.detach().cpu()
                    box_for_viz = boxes.detach().cpu()

                    for i in range(len(img_name)):  # Handle each item in the batch separately
                        visualize_input(
                            image=image_for_viz[i],                  
                            gt_mask=gt_mask_for_viz[i].squeeze(),  
                            box=box_for_viz[i],  
                            pred_mask=pred_mask_for_viz[i].squeeze(),
                            label_id=label_id[i].item(),  
                            image_name=f"{img_name[i]}_QC",
                            model_save_path=self.model_save_path
                        )
                    # del image_for_viz, gt_mask_for_viz, pred_mask_for_viz, box_for_viz
                    # torch.cuda.empty_cache()  # Help free unused memory from PyTorch's cache, although usage should be considered

                    print('Visualization complete!!!')

            if self.use_amp:
                self.scaler.scale(loss).backward()  # Scale the loss for AMP                
                if (self.current_step + 1) % accumulate_steps == 0:  # Only unscale and step the optimizer at specified intervals
                    
                    if self.module_cfg.get('clip_grad', False):
                        self.scaler.unscale_(self.optimizer)  # Unscales the optimizer before clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.module_cfg['clip_grad'])
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                loss.backward()
                if self.module_cfg.get('clip_grad', False):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.module_cfg['clip_grad'])
                if (self.current_step + 1) % accumulate_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
            self.current_step += 1  # Increment the step counter after processing each batch

            if self.module_cfg.get('use_wandb', False):
                wandb_log({'scheduled_lr': self.optimizer.param_groups[0]['lr']})
        # ------------------------------------------------------------------------
        else:
            with torch.no_grad():
                predictions = self.model(image, boxes)
                loss = self.calculate_loss(predictions, gt2D)
                
                # Convert logits to binary predictions
                predictions_binary = torch.sigmoid(predictions) > 0.5
                # Calculate Dice score
                self.dice_metric(y_pred=predictions_binary, y=gt2D)

        if GPUSetup.is_distributed():
            loss = reduce_tensor(loss)  # Reduce loss only if in a distributed environment

        # del predictions, image, gt2D, boxes  # Free variables that are no longer needed
        # torch.cuda.empty_cache()  # Helps free unused memory from PyTorch's cache, although usage should be considered

        return loss.item()
    

    def run_epoch(self, data_loader, epoch, mode='train'):
        self.current_step = 0
        batch_idx = 0  # Initialize batch counter

        log_info(f"Starting {mode} epoch")
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        for batch in data_loader:
            batch_loss = self.process_batch(batch, mode, batch_idx, epoch)
            total_loss += batch_loss
            batch_idx += 1

        average_loss = total_loss / len(data_loader)
        self.shared_losses[mode].append({'epoch': epoch+1, 'loss':average_loss})

        # After all validation batches are processed, compute the Dice score
        if mode == 'val':
            # Aggregate the metric across all batches for the current GPU/process
            dice_score_tensor = self.dice_metric.aggregate()

            if GPUSetup.is_distributed():
                # Reduce the metric across all GPUs
                dist.all_reduce(dice_score_tensor, op=dist.ReduceOp.SUM)
                # Average the sum by the number of GPUs (processes)
                dice_score_tensor /= dist.get_world_size()
            
            # Convert the aggregated Dice score to a Python float
            dice_score = dice_score_tensor.item()

            # Log the global Dice score
            log_info(f"Validation Dice score: {dice_score}")
            self.shared_metrics[mode].append({'epoch': epoch+1, 'dice_score': dice_score})

            if self.module_cfg.get('use_wandb', False):
                wandb_log({f'{mode}_dice': dice_score})
            
            # Reset the metric for the next epoch
            self.dice_metric.reset()

        log_info(f"Finished {mode} epoch with average loss: {average_loss}")
        return average_loss
    

    def check_early_stopping(self, val_loss):
        if val_loss + self.min_delta < self.best_val_loss:
            # Significant improvement found, reset counter
            self.no_improve_epochs = 0
        else:
            # No significant improvement
            self.no_improve_epochs += 1

        # Check if the patience limit has been exceeded
        if self.no_improve_epochs >= self.patience:
            self.early_stopped = True
            should_stop = True
        else:
            should_stop = False

        # Synchronize the early stopping decision across all processes in distributed training
        if GPUSetup.is_distributed():
            # Create a tensor to hold the stop decision on the correct device
            stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int, device=self.device)
            # Broadcast the decision from the main process (assuming rank 0 is main)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

        log_info(f"current num epochs without improvement: {self.no_improve_epochs}")
        return should_stop

    def train(self, num_epochs):
        log_info(f"Starting training for {num_epochs} epochs")
        log_info(f"Early stopping is {'enabled' if self.early_stopping_enabled else 'disabled'}, Patience: {self.patience}, Min delta: {self.min_delta}")

        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')  
        self.no_improve_epochs = 0  # Counter for epochs without improvement

        for epoch in range(self.start_epoch, num_epochs):
            if GPUSetup.is_distributed():
                self.train_loader.sampler.set_epoch(epoch)

            log_info(f"Epoch {epoch+1}/{num_epochs} - Training")
            train_loss = self.run_epoch(self.train_loader, epoch, 'train')

            log_info(f"Epoch {epoch+1}/{num_epochs} - Validation")
            val_loss = self.run_epoch(self.val_loader, epoch, 'val')
            
            # Early stopping check
            if self.check_early_stopping(val_loss):
                log_info(f"Early stopping triggered after {epoch+1} epochs")
                break

            self.best_train_loss, self.best_val_loss = log_and_checkpoint('both', train_loss, val_loss, self.module_cfg, self.model, self.optimizer, epoch, self.model_save_path, self.run_id, self.best_train_loss, self.best_val_loss)

            self.post_epoch_actions(epoch, num_epochs)

        if self.early_stopped: 
            log_info("Training stopped early due to lack of improvement in validation loss.")
        else:
            log_info("Training completed")
        
        self.post_training_summary()

        if self.module_cfg.get('use_wandb', False) and GPUSetup.is_main_process():
            import wandb
            wandb.finish()
        
    @main_process_only
    def post_epoch_actions(self, epoch, num_epochs):
        # Code to plot losses and log/checkpoint after each epoch
        plot_losses(self.shared_losses['train'], self.model_save_path, self.run_id, 'train')
        plot_losses(self.shared_losses['val'], self.model_save_path, self.run_id, 'val')
        plot_metrics(metrics=self.shared_metrics['val'], model_save_path=self.model_save_path, run_id=self.run_id, metric_name='dice_score', mode='val')  

    @main_process_only
    def post_training_summary(self):
        # Code to plot combined losses and save shared losses after training completion
        plot_combined_losses(self.shared_losses['train'], self.shared_losses['val'], self.model_save_path, self.run_id)
        save_losses(self.shared_losses, self.model_save_path, self.run_id)
        save_metrics(self.shared_metrics, self.model_save_path, self.run_id)
        final_checkpoint_conversion(self.module_cfg, self.model_save_path, self.run_id)


