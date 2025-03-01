
import logging
from datetime import datetime
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from monai.metrics import DiceMetric, MeanIoU

from src.finetuning.utils.utils import reduce_tensor
from src.finetuning.utils.logging import main_process_only, log_info, wandb_log
from src.finetuning.utils.gpu_setup import is_distributed, get_world_size, is_main_process
from src.finetuning.utils.checkpointing import log_and_checkpoint, final_checkpoint_conversion

from src.utils import QCV, plot_losses, plot_combined_losses, plot_metrics, save_losses, save_metrics

logger = logging.getLogger(__name__)

class BaseTrainer:
    """
    Base class for training loops, handling:
      - Setup (model, optimizer, scheduler, etc.)
      - Loss computation & metrics
      - The training loop (train, val, early stopping, etc.)
      - Logging and checkpointing
    """
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 loss_fn,
                 train_loader,
                 val_loader,
                 module_cfg,
                 datamodule_cfg,
                 experiment_cfg,
                 run_path,
                 device='cpu',
                 start_epoch=0):
        """
        Parameters
        ----------
        model : torch.nn.Module
            The model to train.
        optimizer : torch.optim.Optimizer
            The optimizer.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler.
        loss_fn : Tuple[callable, callable]
            Typically a tuple of (seg_loss, ce_loss) or something similar.
        train_loader : DataLoader
            The training dataloader.
        val_loader : DataLoader
            The validation dataloader.
        module_cfg : dict
            Contains configuration about training (use_wandb, early_stopping, etc.).
        datamodule_cfg : dict
            Contains dataset-related configuration (batch_size, etc.).
        experiment_cfg : dict
            Contains experiment-level config (descriptions, etc.).
        run_path : str
            The path where logs/checkpoints are saved.
        device : str
            The device to use (e.g., "cuda" or "cpu").
        start_epoch : int
            If resuming from checkpoint, which epoch to start from.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn  # e.g. (seg_loss, ce_loss)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.module_cfg = module_cfg
        self.datamodule_cfg = datamodule_cfg
        self.experiment_cfg = experiment_cfg
        self.run_path = run_path
        self.device = device
        self.start_epoch = start_epoch

        # AMP setup
        self.use_amp = module_cfg.get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Step counter (for gradient accumulation, logging, etc.)
        self.current_step = 0

        # Setup environment (e.g. wandb)
        self.model_save_path, self.run_id = self.setup_experiment_environment()

        # Early stopping
        self.early_stopping_enabled = module_cfg.get('early_stopping', {}).get('enabled', True)
        self.patience = module_cfg.get('early_stopping', {}).get('patience', 5)
        self.min_delta = module_cfg.get('early_stopping', {}).get('min_delta', 0.001)
        self.early_stopped = False

        # Shared losses & metrics across epochs
        self.shared_losses = {'train': [], 'val': []}
        self.shared_metrics = {'train': [], 'val': []}

        # Example metrics (Dice, IoU, etc.)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        # or possibly self.IoU_metric = MeanIoU(...)

    def setup_experiment_environment(self):
        """
        Optionally login/init wandb or other logging frameworks, returning the
        model_save_path and run_id for this training session.
        """
        model_save_path = self.run_path
        run_id = datetime.now().strftime("%Y%m%d-%H%M")

        if self.module_cfg.get('use_wandb', False) and is_main_process():
            import wandb
            wandb.login()
            wandb.init(
                project=self.module_cfg['task_name'],
                group=self.module_cfg.get('group_name', None),
                config={
                    "max_subject_set": self.datamodule_cfg.get('max_subject_set', 'full'),
                    "balanced": self.datamodule_cfg.get('balanced', False),
                    "lr": self.module_cfg.get('optimizer', {}).get('lr', 1e-4),
                    "batch_size": self.datamodule_cfg.get('batch_size', 1),
                    "grad_accum": self.module_cfg.get('grad_accum', 1),
                    "model_type": self.module_cfg.get('model_type', ''),
                    "description": self.experiment_cfg.get('description', ''),
                    "pretrained_weights": self.experiment_cfg.get('pretrained_weights', ''),
                    "image_encoder": self.module_cfg.get('trainable', {}).get('image_encoder', False),
                    "mask_decoder": self.module_cfg.get('trainable', {}).get('mask_decoder', False),
                    "early_stopping": self.module_cfg.get('early_stopping', {}).get('enabled', True),
                    "patience": self.module_cfg.get('early_stopping', {}).get('patience', 5),
                    "bbox_shift": self.datamodule_cfg.get('bbox_shift', 0),
                },
                settings=wandb.Settings(_service_wait=300),
                tags=['train', self.experiment_cfg.get('name', '')],
                name=run_id
            )

        return model_save_path, run_id

    def calculate_loss(self, predictions, gt2D, seg_loss_weight=0.5, ce_loss_weight=0.5):
        """
        Example loss combination. If self.loss_fn is a tuple (seg_loss, ce_loss),
        we combine them here with given weights.
        """
        seg_loss, ce_loss = self.loss_fn
        total_loss = seg_loss_weight * seg_loss(predictions, gt2D) \
                   + ce_loss_weight * ce_loss(predictions, gt2D.float())
        return total_loss
        
    def _maybe_visualize_batch(
        self,
        image: torch.Tensor,
        gt2D: torch.Tensor,
        predictions: torch.Tensor,
        boxes: torch.Tensor,
        label_id: Optional[torch.Tensor],
        img_name: List[str],
        epoch: int,
        batch_idx: int
    ):
        """
        Visualize a batch of data (inputs, ground truth, predictions, etc.)
        Only called if self.module_cfg['visualize'] is True, on rank 0, etc.
        """
        # Typically we only visualize early in training or for certain batches:
        if not self.module_cfg.get('visualize', False):
            return

        # For example, visualize only at epoch==0 and for first few batches:
        if epoch == 0 and batch_idx < 1 and is_main_process():
            with torch.no_grad():
                image_for_viz = image.detach().cpu()
                gt_mask_for_viz = gt2D.detach().cpu()
                pred_mask_for_viz = predictions.detach().cpu()
                boxes_for_viz = boxes.detach().cpu() if boxes is not None else None

                # If label_id is None or not in batch, you can skip it or set a default
                label_id_for_viz = label_id.detach().cpu() if label_id is not None else None

                for i in range(len(img_name)):
                    # visualize_input(
                    QCV.plot_input_with_bbox_and_masks(
                        image=image_for_viz[i],
                        gt_mask=gt_mask_for_viz[i].squeeze(),
                        box=boxes_for_viz[i] if boxes_for_viz is not None else None,
                        pred_mask=pred_mask_for_viz[i].squeeze(),
                        label_id=(
                            label_id_for_viz[i].item() if label_id_for_viz is not None else None
                        ),
                        image_name=f"{img_name[i]}_QC",
                        model_save_path=self.model_save_path
                    )
            print('Visualization complete!!!')


    def process_batch(self, batch, mode='train', batch_idx=0, epoch=0):
        """
        Processes a single batch: forward pass, compute loss, backprop (if train),
        visualize, log, etc.
        """
        # Non-tensor data
        dataset_name = batch.get('dataset_name', None)
        img_name = batch.get('img_name', None)

        # Move everything else to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)

        image = batch['image']   # e.g. [B, 3, H, W]
        gt2D  = batch['gt2D']    # e.g. [B, 1, H, W]
        boxes = batch.get('boxes', None)
        label_id = batch.get('label_id', None)

        if mode == 'train':
            accumulate_steps = self.module_cfg.get('grad_accum', 1)
            # Zero gradients only on the correct accumulation step
            if self.current_step % accumulate_steps == 0:
                self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                predictions = self.model(image, boxes)
                loss = self.calculate_loss(predictions, gt2D) / accumulate_steps

            # <-- Insert visualization call here (only for training, if you want)
            self._maybe_visualize_batch(
                image=image,
                gt2D=gt2D,
                predictions=predictions,
                boxes=boxes,
                label_id=label_id,
                img_name=img_name,
                epoch=epoch,
                batch_idx=batch_idx
            )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Only unscale/step on certain steps
                if (self.current_step + 1) % accumulate_steps == 0:
                    if self.module_cfg.get('clip_grad', False):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.module_cfg['clip_grad']
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                loss.backward()
                if self.module_cfg.get('clip_grad', False):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.module_cfg['clip_grad']
                    )
                if (self.current_step + 1) % accumulate_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

            self.current_step += 1
            if self.module_cfg.get('use_wandb', False):
                wandb_log({'scheduled_lr': self.optimizer.param_groups[0]['lr']})

        else:
            # Validation mode
            with torch.no_grad():
                predictions = self.model(image, boxes)
                loss = self.calculate_loss(predictions, gt2D)

                # Convert logits to binary predictions
                predictions_binary = (torch.sigmoid(predictions) > 0.5)
                # Update the dice metric
                self.dice_metric(y_pred=predictions_binary, y=gt2D)

        if is_distributed():
            # Reduce the loss across all processes
            loss = reduce_tensor(loss)

        return loss.item()

    def run_epoch(self, data_loader, epoch, mode='train'):
        """
        Run a single epoch of either training or validation.
        """
        self.current_step = 0
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        for batch_idx, batch in enumerate(data_loader):
            batch_loss = self.process_batch(batch, mode, batch_idx, epoch)
            total_loss += batch_loss

        average_loss = total_loss / len(data_loader)
        self.shared_losses[mode].append({'epoch': epoch+1, 'loss': average_loss})

        # If in validation, compute aggregated dice score
        if mode == 'val':
            dice_score_tensor = self.dice_metric.aggregate()
            
            if is_distributed():
                dist.all_reduce(dice_score_tensor, op=dist.ReduceOp.SUM)
                dice_score_tensor /= get_world_size()
            
            dice_score = dice_score_tensor.item()
            log_info(f"[Val] Epoch {epoch+1}: dice_score={dice_score:.4f}")
            self.shared_metrics[mode].append({'epoch': epoch+1, 'dice_score': dice_score})
            
            if self.module_cfg.get('use_wandb', False):
                wandb_log({f'{mode}_dice': dice_score})
            
            self.dice_metric.reset()

        log_info(f"Finished {mode} epoch {epoch+1}, avg {mode}_loss={average_loss:.4f}")
        return average_loss

    def check_early_stopping(self, val_loss):
        """
        Evaluate whether to trigger early stopping based on val_loss.
        """
        if val_loss + self.min_delta < self.best_val_loss:
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1

        if self.no_improve_epochs >= self.patience:
            self.early_stopped = True
            should_stop = True
        else:
            should_stop = False

        # Sync across processes
        if is_distributed():
            stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int, device=self.device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = (stop_tensor.item() == 1)

        log_info(f"No improvement epochs: {self.no_improve_epochs}/{self.patience} => Stop? {should_stop}")
        return should_stop

    def train(self, num_epochs):
        """
        Main training loop: run train epochs, val epochs, handle early stopping, etc.
        """
        log_info(f"Starting training for {num_epochs} epochs. Early stopping={self.early_stopping_enabled}, patience={self.patience}, min_delta={self.min_delta}")

        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.no_improve_epochs = 0

        for epoch in range(self.start_epoch, num_epochs):
            if is_distributed():
                self.train_loader.sampler.set_epoch(epoch)

            log_info(f"Epoch {epoch+1}/{num_epochs} - Training")
            train_loss = self.run_epoch(self.train_loader, epoch, mode='train')

            log_info(f"Epoch {epoch+1}/{num_epochs} - Validation")
            val_loss = self.run_epoch(self.val_loader, epoch, mode='val')

            if self.check_early_stopping(val_loss):
                log_info(f"Early stopping triggered at epoch {epoch+1}")
                break

            self.best_train_loss, self.best_val_loss = log_and_checkpoint(
                mode='both',
                train_loss=train_loss,
                val_loss=val_loss,
                module_cfg=self.module_cfg,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                model_save_path=self.model_save_path,
                run_id=self.run_id,
                best_train_loss=self.best_train_loss,
                best_val_loss=self.best_val_loss
            )

            self.post_epoch_actions(epoch, num_epochs)

        if self.early_stopped:
            log_info("Training ended early due to lack of improvement.")
        else:
            log_info("Training completed all epochs.")

        self.post_training_summary()

        if self.module_cfg.get('use_wandb', False) and is_main_process():
            import wandb
            wandb.finish()

    @main_process_only
    def post_epoch_actions(self, epoch, num_epochs):
        """
        Called at the end of each epoch (on the main process) to do any plotting or logging.
        """
        # e.g. plot losses, metrics
        plot_losses(self.shared_losses['train'], self.model_save_path, self.run_id, 'train')
        plot_losses(self.shared_losses['val'],   self.model_save_path, self.run_id, 'val')
        plot_metrics(
            metrics=self.shared_metrics['val'],
            model_save_path=self.model_save_path,
            run_id=self.run_id,
            metric_name='dice_score',
            mode='val'
        )

    @main_process_only
    def post_training_summary(self):
        """
        Called after training finishes (on the main process). Plot final combined losses,
        save logs, convert final checkpoint, etc.
        """
        plot_combined_losses(
            self.shared_losses['train'],
            self.shared_losses['val'],
            self.model_save_path,
            self.run_id
        )
        save_losses(self.shared_losses, self.model_save_path, self.run_id)
        save_metrics(self.shared_metrics, self.model_save_path, self.run_id)
        final_checkpoint_conversion(self.module_cfg, self.model_save_path, self.run_id)



# ------ Structure for Creating Inheriting Trainer Classes for Specific Tasks ------ #

class FinetuningTrainer(BaseTrainer):
    """
    Specialized trainer for finetuning tasks.
    Inherits all logic from BaseTrainer but can override or extend as needed.
    """
    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 loss_fn,
                 train_loader,
                 val_loader,
                 module_cfg,
                 datamodule_cfg,
                 experiment_cfg,
                 run_path,
                 device='cpu',
                 start_epoch=0):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            module_cfg=module_cfg,
            datamodule_cfg=datamodule_cfg,
            experiment_cfg=experiment_cfg,
            run_path=run_path,
            device=device,
            start_epoch=start_epoch
        )
        # If there's any finetuning-specific fields or logic, do it here:
        # e.g., self.some_finetuning_param = module_cfg.get('some_finetuning_param', None)

    # Optionally override any method from the base class if you want different behavior:
    # def calculate_loss(self, predictions, gt2D, seg_loss_weight=0.8, ce_loss_weight=0.2):
    #     return super().calculate_loss(predictions, gt2D, seg_loss_weight, ce_loss_weight)
    #
    # Or override process_batch, run_epoch, etc. as needed.



