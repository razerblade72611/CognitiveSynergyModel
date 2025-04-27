# cognitive_synergy/training/trainer.py
"""
Core Training and Validation Loop Orchestration.

Defines the Trainer class responsible for managing the training process,
including epoch iteration, forward/backward passes, optimization,
validation, logging, and checkpointing. Supports Gradient Accumulation and AMP.
Handles contrastive loss correctly with gradient accumulation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.optimizer import Optimizer
# Use specific scheduler base class for better type hinting if possible
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, SequentialLR, LambdaLR, CosineAnnealingLR, StepLR, MultiStepLR, LinearLR
# Import AMP utilities (should work with ROCm via PyTorch abstraction)
# Update import based on FutureWarning if needed: from torch.amp.cuda import GradScaler, autocast
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Union, List # Added List
import time
import os
import gc # Garbage collector interface
from tqdm import tqdm # Progress bar for nice console output
import logging # Use standard logging

# Assume necessary imports from other project modules if needed
# e.g., from ..utils.logging import setup_logger (or use passed logger)
# e.g., from ..utils.metrics import calculate_accuracy, calculate_retrieval_recall_at_k (or pass metric functions)

# Get a logger instance (either passed in or created here)
logger = logging.getLogger("CognitiveSynergy.Trainer") # Use hierarchical naming

class Trainer:
    """
    Handles the training and validation loops for the Cognitive Synergy Model.

    Manages device placement, forward/backward passes, gradient clipping,
    optimizer and scheduler steps, validation, checkpointing, and logging.
    Includes support for Gradient Accumulation and Automatic Mixed Precision (AMP).
    Modified to handle contrastive loss correctly with gradient accumulation.
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 criterion: nn.Module, # The primary loss function (e.g., ContrastiveAlignmentLoss)
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 scheduler: Optional[_LRScheduler],
                 device: Union[torch.device, str],
                 config: Dict,
                 logger_instance: Optional[logging.Logger] = None): # Pass logger instance
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The model to train (e.g., CognitiveSynergyModel).
            optimizer (Optimizer): The optimizer instance.
            criterion (nn.Module): The loss function module.
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (Optional[DataLoader]): DataLoader for the validation set.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler instance.
            device (Union[torch.device, str]): The device to train on ('cuda', 'cpu', torch.device object).
            config (Dict): Configuration dictionary, expected to contain training settings
                           like 'epochs', 'log_freq', 'checkpoint_dir', 'grad_clip_norm',
                           'gradient_accumulation_steps', 'use_amp', etc.
                           Typically loaded from the YAML config.
            logger_instance (Optional[logging.Logger]): An optional pre-configured logger instance.
                                                        If None, uses the default logger for this module.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device) # Move loss function to device if it has parameters
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.config = config
        # Use passed logger or the module's default logger
        self.logger = logger_instance if logger_instance is not None else logger

        # --- Extract Training Configuration Parameters ---
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 10)
        self.log_freq = config.get('logging', {}).get('log_freq', 100) # Log every N *optimizer* steps
        self.checkpoint_config = config.get('checkpointing', {})
        self.checkpoint_dir = self.checkpoint_config.get('checkpoint_dir', './checkpoints')
        self.save_freq = self.checkpoint_config.get('save_freq', 1) # Save every N epochs
        self.save_best_only = self.checkpoint_config.get('save_best_only', False)
        self.save_last = self.checkpoint_config.get('save_last', True) # Option to save last epoch ckpt
        self.grad_clip_norm = train_config.get('grad_clip_norm', None) # Optional gradient clipping max norm

        # --- Gradient Accumulation ---
        self.gradient_accumulation_steps = train_config.get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps < 1:
            self.logger.warning(f"gradient_accumulation_steps ({self.gradient_accumulation_steps}) is less than 1. Setting to 1.")
            self.gradient_accumulation_steps = 1
        # --- Check if accumulation is needed for contrastive loss ---
        if self.train_loader.batch_size == 1 and self.gradient_accumulation_steps <= 1:
             # Note: ContrastiveLoss class itself handles batch_size=1, but won't learn.
             self.logger.warning(f"Training DataLoader batch_size is 1 and gradient_accumulation_steps is {self.gradient_accumulation_steps}. "
                                 "Contrastive loss requires an effective batch size > 1 to learn properly. "
                                 "Ensure trainer accumulates embeddings or increase batch_size/accumulation_steps.")
        elif self.train_loader.batch_size * self.gradient_accumulation_steps <= 1:
             self.logger.warning(f"Effective batch size ({train_loader.batch_size * self.gradient_accumulation_steps}) is <= 1. "
                                 "Contrastive loss requires an effective batch size > 1 to learn properly.")


        # --- Mixed Precision (AMP) ---
        self.use_amp = train_config.get('use_amp', False)
        # Initialize GradScaler
        # Handle potential deprecation warning for older torch versions if needed
        # self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp) # Newer syntax
        self.scaler = GradScaler(enabled=self.use_amp) # Older syntax, often still works


        # --- Internal State ---
        self.current_epoch = 0 # Tracks the current epoch (0-based)
        self.global_step = 0 # Tracks the total number of *optimizer* steps taken
        self.best_val_metric = float('inf') # Use 'inf' for loss, '-inf' for accuracy
        # Define metric for saving best checkpoint (can be overridden by loaded checkpoint)
        self.validation_metric_name = config.get('validation', {}).get('best_metric', 'val/loss_epoch')
        self.validation_metric_mode = config.get('validation', {}).get('best_metric_mode', 'min') # 'min' for loss, 'max' for accuracy
        if self.validation_metric_mode == 'max':
            self.best_val_metric = float('-inf')

        self.logger.info("Trainer initialized.")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Epochs: {self.epochs}")
        self.logger.info(f"  Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        self.logger.info(f"  DataLoader Batch Size: {train_loader.batch_size}")
        self.logger.info(f"  Effective Batch Size: {train_loader.batch_size * self.gradient_accumulation_steps}")
        self.logger.info(f"  Mixed Precision Training (AMP): {'Enabled' if self.use_amp else 'Disabled'}")
        self.logger.info(f"  Log Frequency: {self.log_freq} optimizer steps")
        self.logger.info(f"  Checkpoint Directory: {self.checkpoint_dir}")
        self.logger.info(f"  Save Frequency: Every {self.save_freq} epochs")
        self.logger.info(f"  Save Best Only: {self.save_best_only} (based on {self.validation_metric_name} in '{self.validation_metric_mode}' mode)")
        self.logger.info(f"  Save Last Checkpoint: {self.save_last}")
        if self.grad_clip_norm:
            self.logger.info(f"  Gradient Clipping Max Norm: {self.grad_clip_norm}")

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Moves batch tensors to the configured device."""
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device, non_blocking=True)
            # else: # Optionally handle non-tensor data if needed (e.g., list of strings for metrics)
            #     prepared_batch[key] = value
        return prepared_batch

    def _is_step_scheduler(self) -> bool:
        """Checks if the scheduler should be stepped per optimizer step."""
        step_based = (LambdaLR, CosineAnnealingLR, LinearLR)
        # Check if SequentialLR contains any step-based schedulers
        if isinstance(self.scheduler, SequentialLR):
            # This check assumes the internal schedulers list is accessible and indicative
            # A more robust check might be needed depending on PyTorch version/scheduler complexity
             return any(isinstance(s, step_based) for s in self.scheduler._schedulers)
        return isinstance(self.scheduler, step_based)


    def _is_epoch_scheduler(self) -> bool:
        """Checks if the scheduler should be stepped per epoch."""
        epoch_based = (StepLR, MultiStepLR)
        # Check if SequentialLR contains only epoch-based schedulers (less common for warmup)
        if isinstance(self.scheduler, SequentialLR):
             return all(isinstance(s, epoch_based) or isinstance(s, ReduceLROnPlateau) for s in self.scheduler._schedulers) # <<< FIXED
        return isinstance(self.scheduler, epoch_based) or isinstance(self.scheduler, ReduceLROnPlateau)


    def _train_epoch(self):
        """Runs one epoch of training, accumulating embeddings for contrastive loss."""
        self.model.train()
        total_epoch_loss = 0.0 # Sum of losses from each optimizer step
        num_optimizer_steps = 0 # Count actual optimizer steps this epoch

        # ---> NEW: Lists to accumulate embeddings and losses over accumulation steps <---
        accumulated_vision_embs = []
        accumulated_language_embs = []
        accumulated_batch_losses = [] # Store loss from each opt step for logging avg

        epoch_start_time = time.time()
        num_batches = len(self.train_loader)
        pbar = tqdm(enumerate(self.train_loader), total=num_batches,
                    desc=f"Epoch {self.current_epoch+1}/{self.epochs} [Train]", leave=False)

        # Zero gradients once before the loop
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in pbar:
            batch_start_time = time.time()
            try:
                prepared_batch = self._prepare_batch(batch)
                image_input = prepared_batch['image']
                input_ids = prepared_batch['input_ids']
                attention_mask = prepared_batch['attention_mask']
            except Exception as e:
                self.logger.error(f"Error preparing batch at step {step}: {e}. Skipping batch.", exc_info=True)
                continue

            # --- Forward Pass ---
            try:
                # Run forward pass under autocast context
                with autocast(enabled=self.use_amp):
                    model_outputs = self.model(image_input, input_ids, attention_mask)
                    vision_emb = model_outputs.get('projected_vision_cls')
                    language_emb = model_outputs.get('projected_language_cls')

                    if vision_emb is None or language_emb is None:
                        raise KeyError("Missing projected embeddings in model output.")

                # --- Accumulate DETACHED embeddings for loss calculation later ---
                accumulated_vision_embs.append(vision_emb)
                accumulated_language_embs.append(language_emb)

            except Exception as e:
                self.logger.error(f"Error during forward pass at step {step}: {e}", exc_info=True)
                # If forward fails, clear accumulators for this cycle and skip opt step
                accumulated_vision_embs.clear()
                accumulated_language_embs.clear()
                # Ensure gradients are clear if an error happens mid-cycle
                # (zero_grad happens before opt step anyway)
                continue # Skip to next mini-batch

            # --- Check if it's time for an Optimizer Step ---
            is_accumulation_step = (step + 1) % self.gradient_accumulation_steps == 0
            is_last_step = (step + 1) == num_batches

            if is_accumulation_step or is_last_step:
                # --- Loss Calculation on Accumulated Batch ---
                if not accumulated_vision_embs or not accumulated_language_embs:
                    self.logger.warning(f"Skipping optimizer step at step {step} (Global {self.global_step}) due to empty accumulated embeddings (likely previous forward error).")
                    continue # Skip if accumulation is empty

                try:
                    # Use autocast for the loss calculation as well if using AMP
                    with autocast(enabled=self.use_amp):
                        # Concatenate collected embeddings
                        vision_batch = torch.cat(accumulated_vision_embs, dim=0)
                        language_batch = torch.cat(accumulated_language_embs, dim=0)

                        # Ensure requires_grad is true for loss input if needed (should be handled by autocast context)
                        # Calculate loss ONCE on the effective batch
                        loss = self.criterion(vision_batch, language_batch)

                    # --- Loss Normalization & Backward ---
                    # Normalize loss by accumulation steps before scaling
                    normalized_loss = loss / self.gradient_accumulation_steps
                    current_loss_value = loss.item() # For logging/tracking
                    accumulated_batch_losses.append(current_loss_value) # Track loss for this opt step
                    total_epoch_loss += current_loss_value # Add loss of this opt step to epoch total

                    # Scaled backward pass
                    self.scaler.scale(normalized_loss).backward()

                except Exception as e:
                    self.logger.error(f"Error during loss/backward at opt step (data step {step}, Global {self.global_step}): {e}", exc_info=True)
                    # Clear accumulators as this cycle failed
                    accumulated_vision_embs.clear()
                    accumulated_language_embs.clear()
                    self.optimizer.zero_grad(set_to_none=True) # Clear potentially bad grads
                    continue # Skip opt step

                # --- Optimizer Step, Scheduler Step, Logging ---
                try:
                    # Gradient Clipping (Optional)
                    if self.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer) # Unscale first
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                    # Optimizer Step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    # Scheduler Step (if step-based)
                    if self.scheduler and self._is_step_scheduler():
                        self.scheduler.step()

                    # Zero Gradients for the next cycle
                    self.optimizer.zero_grad(set_to_none=True)

                    # Increment counts
                    self.global_step += 1
                    num_optimizer_steps += 1

                    # Logging (now based on global optimizer step)
                    if self.global_step % self.log_freq == 0:
                        # Average loss over the batches included in this optimizer step
                        avg_loss_for_step = sum(accumulated_batch_losses) / len(accumulated_batch_losses) if accumulated_batch_losses else float('nan')
                        log_data = {
                            'train/loss_opt_step_avg': avg_loss_for_step,
                            'train/lr': self.optimizer.param_groups[0]['lr'],
                            'train/amp_scale': self.scaler.get_scale() if self.use_amp else -1.0,
                            # Batch time is tricky, maybe log time per opt step
                            'train/opt_step_time_ms': (time.time() - batch_start_time) * 1000, # Rough time for last mini-batch in cycle
                            'epoch': self.current_epoch + 1
                        }
                        if hasattr(self.logger, 'log_metrics'):
                            self.logger.log_metrics(log_data, step=self.global_step)
                        else:
                            self.logger.info(f"Step: {self.global_step:>7} | " + " | ".join([f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log_data.items()]))
                        accumulated_batch_losses.clear() # Clear losses after logging

                except Exception as e:
                    self.logger.error(f"Error during optimizer/scaler/scheduler step at Global Step {self.global_step}: {e}", exc_info=True)
                    # Clear accumulators if step failed
                    accumulated_vision_embs.clear()
                    accumulated_language_embs.clear()
                    self.optimizer.zero_grad(set_to_none=True) # Ensure grads cleared
                    continue # Skip to next data batch

                # --- Clear accumulated embeddings ---
                accumulated_vision_embs.clear()
                accumulated_language_embs.clear()

            # --- Update progress bar ---
            last_opt_step_loss = accumulated_batch_losses[-1] if accumulated_batch_losses else float('nan')
            pbar.set_postfix(
                loss=f"{last_opt_step_loss:.4f}", # Loss from last successful opt step
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                step=f"{step+1}/{num_batches}"
            )

        # --- End of Training Epoch ---
        avg_epoch_loss = total_epoch_loss / num_optimizer_steps if num_optimizer_steps > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        log_data_epoch = {
            'train/loss_epoch': avg_epoch_loss, # Average loss per optimizer step
            'train/epoch_time_sec': epoch_time,
            'epoch': self.current_epoch + 1
        }
        if hasattr(self.logger, 'log_metrics'):
            self.logger.log_metrics(log_data_epoch, step=self.global_step)
        else:
            self.logger.info(f"Epoch {self.current_epoch+1} Train Summary: Avg Opt Step Loss: {avg_epoch_loss:.5f}, Time: {epoch_time:.2f}s")

        # Step epoch-based schedulers (that are not ReduceLROnPlateau)
        if self.scheduler and self._is_epoch_scheduler() and not isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step()

        # --- Optional End-of-Epoch Memory Clearing ---
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()


    @torch.no_grad()
    def _validate_epoch(self) -> Optional[float]:
        """Runs one epoch of validation and returns the primary validation metric."""
        if self.val_loader is None:
            self.logger.info("Validation loader not provided, skipping validation.")
            return None

        self.model.eval()
        total_val_loss = 0.0
        num_samples = 0
        epoch_start_time = time.time()

        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader),
                    desc=f"Epoch {self.current_epoch+1}/{self.epochs} [Validate]", leave=False)

        # Store embeddings for potential contrastive loss calculation on the whole val set
        # This might cause OOM on validation if the val set is large.
        # Alternatively, calculate loss per batch as done before.
        # Let's stick to per-batch calculation for simplicity and memory safety.
        # all_val_vision_embs = []
        # all_val_language_embs = []

        for step, batch in pbar:
            batch_start_time = time.time()
            try:
                prepared_batch = self._prepare_batch(batch)
                image_input = prepared_batch['image']
                input_ids = prepared_batch['input_ids']
                attention_mask = prepared_batch['attention_mask']
                current_batch_size = image_input.shape[0]
                num_samples += current_batch_size
            except Exception as e:
                self.logger.error(f"Error preparing validation batch {step}: {e}. Skipping.", exc_info=True)
                continue

            # --- Forward Pass ---
            # Use autocast during validation ONLY if needed AND if criterion benefits
            # Standard practice is often to validate in FP32 unless memory is extremely tight
            # or if AMP instability during training requires validating with it too.
            # Let's keep validation in FP32 for simplicity unless issues arise.
            try:
                # with autocast(enabled=self.use_amp): # Usually not needed for validation
                model_outputs = self.model(image_input, input_ids, attention_mask)
                vision_emb = model_outputs.get('projected_vision_cls')
                language_emb = model_outputs.get('projected_language_cls')

                if vision_emb is None or language_emb is None:
                    raise KeyError("Missing projected embeddings in validation output.")

                # Calculate loss per batch
                loss = self.criterion(vision_emb, language_emb)
                total_val_loss += loss.item() * current_batch_size # Weighted by batch size

            except Exception as e:
                 self.logger.error(f"Error during validation forward/loss at step {step}: {e}", exc_info=True)
                 continue

            batch_time = time.time() - batch_start_time
            pbar.set_postfix(loss=f"{loss.item():.4f}", time=f"{batch_time:.3f}s")

        # --- End of Validation Epoch ---
        avg_val_loss = total_val_loss / num_samples if num_samples > 0 else 0.0
        epoch_time = time.time() - epoch_start_time

        log_data_epoch = {
            'val/loss_epoch': avg_val_loss,
            'val/epoch_time_sec': epoch_time,
            'epoch': self.current_epoch + 1
        }

        if hasattr(self.logger, 'log_metrics'):
             self.logger.log_metrics(log_data_epoch, step=self.global_step)
        else:
             self.logger.info(f"Epoch {self.current_epoch+1} Validation Summary: Avg Loss: {avg_val_loss:.5f}, Time: {epoch_time:.2f}s")

        # Step ReduceLROnPlateau scheduler if used
        if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
             self.scheduler.step(avg_val_loss)

        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        primary_metric_value = log_data_epoch.get(self.validation_metric_name, None)
        if primary_metric_value is None:
             self.logger.warning(f"Primary validation metric '{self.validation_metric_name}' not found in logs.")
        return primary_metric_value


    def save_checkpoint(self, is_best: bool = False, filename_prefix: str = "checkpoint"):
        """Saves model, optimizer, scheduler state, scaler state, and training progress."""
        if not self.checkpoint_dir:
            self.logger.warning("Checkpoint directory not set. Skipping checkpoint saving.")
            return

        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'validation_metric_name': self.validation_metric_name,
            'validation_metric_mode': self.validation_metric_mode,
            'config': self.config
        }
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp:
            state['scaler_state_dict'] = self.scaler.state_dict()

        filename = f"{filename_prefix}_epoch_{self.current_epoch+1}_step_{self.global_step}.pth.tar"
        filepath = os.path.join(self.checkpoint_dir, filename)

        try:
            torch.save(state, filepath)
            self.logger.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint to {filepath}: {e}", exc_info=True)

        if is_best:
            best_filename = f"{filename_prefix}_best.pth.tar"
            best_filepath = os.path.join(self.checkpoint_dir, best_filename)
            try:
                torch.save(state, best_filepath)
                self.logger.info(f"Best checkpoint saved to {best_filepath} (Metric: {self.best_val_metric:.5f})")
            except Exception as e:
                self.logger.error(f"Error saving best checkpoint to {best_filepath}: {e}", exc_info=True)


    def load_checkpoint(self, filepath: str):
        """Loads state from a checkpoint file to resume training."""
        if not os.path.exists(filepath):
            self.logger.warning(f"Checkpoint file not found at {filepath}. Cannot resume.")
            return False

        self.logger.info(f"Loading checkpoint from {filepath}...")
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
        except Exception as e:
             self.logger.error(f"Error loading checkpoint file {filepath}: {e}", exc_info=True)
             return False

        try:
            state_dict = checkpoint['model_state_dict']
            is_ddp_state = all(k.startswith('module.') for k in state_dict.keys())
            if is_ddp_state and not isinstance(self.model, nn.parallel.DistributedDataParallel):
                 self.logger.info("  Detected DDP prefix ('module.') in checkpoint state_dict but model is not DDP. Removing prefix.")
                 state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
            elif not is_ddp_state and isinstance(self.model, nn.parallel.DistributedDataParallel):
                 self.logger.warning("  Model is DDP but checkpoint state_dict lacks 'module.' prefix. Loading as is, may fail if not intended.")

            load_result = self.model.load_state_dict(state_dict, strict=True) # Use strict=True for resuming usually
            self.logger.info(f"  Model state loaded: {load_result}")

        except Exception as e:
             self.logger.error(f"Error loading model state_dict: {e}. Trying with strict=False.", exc_info=True)
             try:
                  load_result = self.model.load_state_dict(state_dict, strict=False)
                  self.logger.warning(f"  Model state partially loaded (strict=False): {load_result}")
             except Exception as e2:
                  self.logger.error(f"  Failed to load model state_dict even with strict=False: {e2}", exc_info=True)
                  return False # Fail loading if model state is critical


        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                self.logger.info("  Optimizer state loaded.")
            except Exception as e:
                 self.logger.warning(f"Could not load optimizer state_dict: {e}. Optimizer state reset.", exc_info=True)
        else:
             self.logger.warning("Optimizer state_dict not found in checkpoint.")


        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("  Scheduler state loaded.")
            except Exception as e:
                 self.logger.warning(f"Could not load scheduler state_dict: {e}. Scheduler state reset.", exc_info=True)
        elif self.scheduler is not None:
             self.logger.warning("Scheduler state_dict not found in checkpoint.")


        if self.use_amp and 'scaler_state_dict' in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.logger.info("  AMP Scaler state loaded.")
            except Exception as e:
                 self.logger.warning(f"Could not load AMP scaler state_dict: {e}. Scaler state reset.", exc_info=True)
        elif self.use_amp:
             self.logger.warning("AMP scaler state_dict not found in checkpoint (required as use_amp=True). Scaler state reset.")


        # Load Training Progress
        self.current_epoch = checkpoint.get('epoch', -1) + 1
        self.global_step = checkpoint.get('global_step', 0)
        default_best = float('inf') if self.validation_metric_mode == 'min' else float('-inf')
        self.best_val_metric = checkpoint.get('best_val_metric', default_best)
        self.validation_metric_name = checkpoint.get('validation_metric_name', self.validation_metric_name)
        self.validation_metric_mode = checkpoint.get('validation_metric_mode', self.validation_metric_mode)


        self.logger.info(f"Checkpoint loaded successfully. Resuming from start of Epoch {self.current_epoch + 1}, Global Step {self.global_step}.")
        self.logger.info(f"  Best validation metric ({self.validation_metric_name}) loaded: {self.best_val_metric:.5f}")
        return True


    def train(self, resume_checkpoint_path: Optional[str] = None):
        """Runs the full training loop over all configured epochs."""
        # --- Attempt to Resume Training ---
        resume_path = resume_checkpoint_path if resume_checkpoint_path is not None else self.checkpoint_config.get('resume_from_checkpoint', None)
        if resume_path:
            if resume_path == 'latest':
                try:
                    self.logger.info(f"Attempting to find latest checkpoint in {self.checkpoint_dir}...")
                    checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth.tar')]
                    if checkpoints:
                        checkpoints.sort(key=lambda f: int(f.split('_epoch_')[1].split('_')[0]), reverse=True)
                        latest_checkpoint = os.path.join(self.checkpoint_dir, checkpoints[0])
                        self.logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                        self.load_checkpoint(latest_checkpoint)
                    else:
                        self.logger.warning("resume_from_checkpoint set to 'latest', but no checkpoints found. Starting from scratch.")
                except Exception as e:
                    self.logger.error(f"Error finding latest checkpoint: {e}. Starting from scratch.", exc_info=True)
            else:
                 self.load_checkpoint(resume_path)

        self.logger.info(f"--- Starting Training from Epoch {self.current_epoch + 1} ---")
        start_total_time = time.time()

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_label = f"Epoch {self.current_epoch + 1}/{self.epochs}"
            self.logger.info(f"\n===== {epoch_label} =====")

            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch()

            val_freq = self.config.get('validation', {}).get('val_freq', 1)
            run_validation = (self.val_loader is not None) and \
                             ((self.current_epoch + 1) % val_freq == 0)

            current_val_metric = None
            if run_validation:
                current_val_metric = self._validate_epoch()
            elif self.val_loader is not None:
                 self.logger.info(f"Skipping validation for epoch {self.current_epoch + 1} (val_freq={val_freq}).")

            # --- Checkpointing ---
            is_best = False
            save_epoch_checkpoint = (self.save_freq > 0 and (self.current_epoch + 1) % self.save_freq == 0)

            if current_val_metric is not None:
                 if self.validation_metric_mode == 'min':
                     if current_val_metric < self.best_val_metric:
                         self.best_val_metric = current_val_metric
                         is_best = True
                 elif self.validation_metric_mode == 'max':
                     if self.best_val_metric == float('-inf') or current_val_metric > self.best_val_metric:
                         self.best_val_metric = current_val_metric
                         is_best = True

                 if is_best:
                      self.logger.info(f"*** New best validation metric ({self.validation_metric_name}): {self.best_val_metric:.5f} ***")
                      # Save regardless of save_best_only if it's the best
                      self.save_checkpoint(is_best=True)

            # Save epoch checkpoint if freq met AND (it's not best OR we are not saving best only)
            if save_epoch_checkpoint and (not is_best or not self.save_best_only):
                 # Avoid saving duplicate if it was already saved as best
                 if not is_best: # Only save regular checkpoint if it wasn't just saved as best
                      self.save_checkpoint(is_best=False)


        # --- End of Training ---
        total_training_time = time.time() - start_total_time
        self.logger.info(f"\n--- Training Finished After {self.epochs} Epochs ---")
        self.logger.info(f"Total Training Time: {total_training_time / 3600:.2f} hours ({total_training_time:.1f} seconds)")
        if self.val_loader: # Only log best metric if validation was performed
             self.logger.info(f"Best Validation Metric ({self.validation_metric_name}): {self.best_val_metric:.5f}")

        if self.save_last:
             self.logger.info("Saving final checkpoint...")
             self.save_checkpoint(is_best=False, filename_prefix="checkpoint_last")

        if hasattr(self.logger, 'close'):
            self.logger.close()
        elif self.config.get('logging', {}).get('use_wandb', False):
             try:
                  # Check if wandb was actually imported and initialized
                  if 'wandb' in sys.modules and sys.modules['wandb'] is not None and hasattr(sys.modules['wandb'], 'run') and sys.modules['wandb'].run is not None:
                       print("Finishing Weights & Biases run...")
                       sys.modules['wandb'].finish()
             except Exception as e:
                  self.logger.error(f"Error finishing wandb run: {e}")

        self.logger.info("Trainer finished.")
