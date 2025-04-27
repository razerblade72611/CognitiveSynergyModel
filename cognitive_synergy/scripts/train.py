# cognitive_synergy/scripts/train.py
"""
Main Training Script for the Cognitive Synergy Model.

This script orchestrates the entire training process:
1. Parses command-line arguments (config path, overrides).
2. Loads configuration from YAML file.
3. Sets up logging (console, file, WandB).
4. Sets random seeds for reproducibility.
5. Initializes dataset, transforms, and dataloaders.
6. Initializes the Cognitive Synergy Model.
7. Initializes the loss function, optimizer, and scheduler.
8. Initializes the Trainer class.
9. Starts the training loop.
"""

import argparse
import yaml # PyYAML for loading config: pip install pyyaml
import os
import sys # For exiting on critical errors
import pprint # For pretty printing config
import torch
from typing import Dict, Any, Optional

# --- Import project components ---
# Use try-except blocks for robustness if run before all modules exist
try:
    # Assuming the script is run from the project root (e.g., My_Last_Project/Something_new/)
    # and the cognitive_synergy package is in the Python path (e.g., via PYTHONPATH or installation)
    from cognitive_synergy.models import CognitiveSynergyModel
    from cognitive_synergy.data.datasets import ContrastiveImageTextDataset # Or a factory function
    from cognitive_synergy.data.transforms import get_image_transform, TextTransform
    from cognitive_synergy.data.dataloaders import create_dataloader
    from cognitive_synergy.training.losses import get_loss_function, ContrastiveAlignmentLoss # Use factory or specific class
    from cognitive_synergy.training.optimizers import create_optimizer, create_scheduler
    from cognitive_synergy.training.trainer import Trainer
    from cognitive_synergy.utils.logging import setup_logger
    from cognitive_synergy.utils.misc import set_seed, save_config
except ImportError as e:
    print(f"Error importing project modules in scripts/train.py: {e}")
    print("Please ensure all required modules exist and the script is run such that the 'cognitive_synergy' package is importable.")
    print("Example: Run from the parent directory of 'cognitive_synergy/' or install the package.")
    # Define dummy classes/functions to allow script structure definition if needed for linting
    class Dummy: pass
    def dummy_func(*args, **kwargs): pass
    CognitiveSynergyModel = ContrastiveImageTextDataset = Trainer = Dummy
    get_image_transform = TextTransform = create_dataloader = get_loss_function = ContrastiveAlignmentLoss = create_optimizer = create_scheduler = setup_logger = set_seed = save_config = dummy_func
    # Exit if imports fail in a real scenario
    sys.exit(f"Failed to import necessary modules: {e}")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the Cognitive Synergy Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file (e.g., configs/base_config.yaml)."
    )
    # --- Optional Command-Line Overrides ---
    # Example: Override batch size
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size from config.")
    # Example: Override epochs
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs from config.")
    # Example: Override learning rate
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config.")
    # Example: Override device
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., 'cuda:0', 'cpu').")
    # Example: Override checkpoint resume path
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training (overrides config).")
    # Example: Override WandB usage
    parser.add_argument("--use_wandb", action='store_true', default=None, help="Force enable WandB logging.")
    parser.add_argument("--no_wandb", action='store_false', dest='use_wandb', help="Force disable WandB logging.")


    args = parser.parse_args()
    return args

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty config file case
             raise ValueError(f"Configuration file {config_path} is empty or invalid.")
        print(f"Loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        print(f"Error loading configuration file {config_path}: {e}")
        raise

def override_config_with_args(config: Dict, args: argparse.Namespace):
     """Overrides configuration dictionary values with command-line arguments if provided."""
     if args.batch_size is not None:
          config['training']['batch_size'] = args.batch_size
          print(f"  Overriding training.batch_size with CLI arg: {args.batch_size}")
     if args.epochs is not None:
          config['training']['epochs'] = args.epochs
          print(f"  Overriding training.epochs with CLI arg: {args.epochs}")
     if args.lr is not None:
          config['training']['learning_rate'] = args.lr
          print(f"  Overriding training.learning_rate with CLI arg: {args.lr}")
     if args.device is not None:
          config['device'] = args.device
          print(f"  Overriding device with CLI arg: {args.device}")
     if args.resume is not None:
          config['checkpointing']['resume_from_checkpoint'] = args.resume
          print(f"  Overriding checkpointing.resume_from_checkpoint with CLI arg: {args.resume}")
     if args.use_wandb is not None: # Handles both --use_wandb and --no_wandb
          config['logging']['use_wandb'] = args.use_wandb
          print(f"  Overriding logging.use_wandb with CLI arg: {args.use_wandb}")
     return config


def main():
    """Main function to setup and run training."""
    args = parse_args()
    config = load_config(args.config)
    config = override_config_with_args(config, args) # Apply CLI overrides

    # --- Setup ---
    # Pretty print the final effective configuration
    print("\n--- Effective Configuration ---")
    pprint.pprint(config, indent=2)
    print("-----------------------------\n")

    # Set random seed early
    seed = config.get('seed', 42)
    set_seed(seed)

    # Setup Logger (handles console, file, wandb based on config)
    # Pass the final config to the logger setup
    logger = setup_logger(config=config)
    logger.info("Starting training script...")
    logger.info(f"Using configuration file: {args.config}")
    if any(vars(args).values()): # Check if any CLI args were used
         logger.info(f"Applied command-line overrides: {vars(args)}")
    logger.info(f"Random seed set to: {seed}")

    # Setup Device
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        logger.warning(f"CUDA device '{device_str}' requested but CUDA not available. Switching to CPU.")
        device = torch.device('cpu')
    else:
        try:
             device = torch.device(device_str)
             # Test device availability briefly
             _ = torch.tensor([1]).to(device)
        except Exception as e:
             logger.error(f"Could not set device to '{device_str}'. Error: {e}. Falling back to CPU.", exc_info=True)
             device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Setting up data loaders...")
    data_config = config.get('data', {})
    train_config = config.get('training', {})
    val_config = config.get('validation', {})

    # Validate required data config keys
    required_data_keys = ['train_manifest', 'val_manifest']
    for key in required_data_keys:
        if key not in data_config or not data_config[key]:
             logger.error(f"Missing or empty required key in data config: '{key}'. Please specify dataset paths in the config file.")
             sys.exit(1)

    # Create transforms
    # Note: Tokenizer might be loaded within TextTransform or passed in
    # Assuming TextTransform loads tokenizer based on language backbone name
    try:
        text_transform = TextTransform(
            tokenizer_name=config.get('backbones', {}).get('language', {}).get('model_name', 'bert-base-uncased'),
            max_length=data_config.get('max_text_length', 128)
        )
        train_image_transform = get_image_transform(
            image_size=data_config.get('image_size', 224),
            is_train=True
        )
        val_image_transform = get_image_transform(
            image_size=data_config.get('image_size', 224),
            is_train=False
        )
    except Exception as e:
         logger.error(f"Error creating data transforms: {e}", exc_info=True)
         sys.exit(1)

    # Create datasets
    try:
        train_dataset = ContrastiveImageTextDataset(
            manifest_path=data_config['train_manifest'],
            image_transform=train_image_transform,
            text_transform=text_transform,
            image_root=data_config.get('image_root', None)
        )
        val_dataset = ContrastiveImageTextDataset(
            manifest_path=data_config['val_manifest'],
            image_transform=val_image_transform,
            text_transform=text_transform,
            image_root=data_config.get('image_root', None)
        )
    except Exception as e:
         logger.error(f"Error creating datasets: {e}", exc_info=True)
         sys.exit(1)

    # Create dataloaders
    # TODO: Add distributed training args (use_distributed, world_size, rank) if needed
    try:
        train_loader = create_dataloader(
            dataset=train_dataset,
            batch_size=train_config.get('batch_size', 32),
            num_workers=train_config.get('num_workers', 0),
            shuffle=True, # Shuffle training data
            pin_memory=train_config.get('pin_memory', True),
            drop_last=train_config.get('drop_last_batch', False),
            # collate_fn= # Add custom collate if needed
        )
        val_loader = create_dataloader(
            dataset=val_dataset,
            batch_size=val_config.get('batch_size', train_config.get('batch_size', 64)), # Use separate val batch size
            num_workers=train_config.get('num_workers', 0),
            shuffle=False, # No shuffling for validation
            pin_memory=train_config.get('pin_memory', True),
            drop_last=False,
            # collate_fn=
        )
    except Exception as e:
         logger.error(f"Error creating dataloaders: {e}", exc_info=True)
         sys.exit(1)

    logger.info(f"Train DataLoader: {len(train_loader)} batches, Batch Size: {train_config.get('batch_size', 32)}")
    logger.info(f"Validation DataLoader: {len(val_loader)} batches, Batch Size: {val_config.get('batch_size', train_config.get('batch_size', 64))}")

    # --- Model Initialization ---
    logger.info("Initializing model...")
    try:
        model = CognitiveSynergyModel(config=config)
    except Exception as e:
         logger.error(f"Error initializing CognitiveSynergyModel: {e}", exc_info=True)
         sys.exit(1)

    # --- Loss, Optimizer, Scheduler ---
    logger.info("Initializing loss, optimizer, and scheduler...")
    try:
        criterion = get_loss_function(config).to(device) # Using factory
        optimizer = create_optimizer(model.parameters(), config.get('training', {}))

        # --- MODIFIED: Correct steps_per_epoch calculation ---
        # Get gradient accumulation steps from config
        accumulation_steps = config.get('training', {}).get('gradient_accumulation_steps', 1)
        if accumulation_steps < 1: accumulation_steps = 1 # Ensure positive
        # Calculate steps per epoch based on optimizer steps
        steps_per_epoch = len(train_loader) // accumulation_steps
        if steps_per_epoch == 0: # Handle case where dataloader is smaller than accumulation steps
            steps_per_epoch = 1
            logger.warning(f"DataLoader length ({len(train_loader)}) is less than gradient_accumulation_steps ({accumulation_steps}). Setting steps_per_epoch=1.")
        # ---------------------------------------------------------

        scheduler = create_scheduler(optimizer, config.get('training', {}), steps_per_epoch=steps_per_epoch) # Pass correct value
    except Exception as e:
        logger.error(f"Error initializing loss/optimizer/scheduler: {e}", exc_info=True)
        sys.exit(1)

    # --- Trainer Initialization ---
    logger.info("Initializing Trainer...")
    try:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            device=device,
            config=config, # Pass the full effective config to the trainer
            logger_instance=logger # Pass the configured logger
        )
    except Exception as e:
         logger.error(f"Error initializing Trainer: {e}", exc_info=True)
         sys.exit(1)

    # --- Save Final Config ---
    # Save the effective config (including overrides) to the checkpoint dir for reproducibility
    try:
        # Ensure checkpoint dir exists (Trainer also does this, but good practice here too)
        checkpoint_dir = trainer.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        config_save_path = os.path.join(checkpoint_dir, "effective_config.yaml")
        save_config(config, config_save_path)
        logger.info(f"Effective configuration saved to {config_save_path}")
    except Exception as e:
         logger.warning(f"Could not save effective configuration: {e}")


    # --- Start Training ---
    logger.info("Starting training process...")
    try:
        # Resume path is determined by CLI arg (handled in override) or config
        resume_path_final = config.get('checkpointing', {}).get('resume_from_checkpoint', None)
        trainer.train(resume_checkpoint_path=resume_path_final) # Pass final resume path to trainer
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt). Saving final state...")
        # Optionally save final state on interrupt
        trainer.save_checkpoint(filename_prefix="interrupt_checkpoint")
        logger.info("Interrupt checkpoint saved (if possible). Exiting.")
        sys.exit(0) # Clean exit
    except Exception as e:
         logger.error("An error occurred during training.", exc_info=True)
         # Optionally save final state on error
         logger.info("Attempting to save error checkpoint...")
         trainer.save_checkpoint(filename_prefix="error_checkpoint")
         sys.exit(1) # Exit with error code
    finally:
        # --- Cleanup ---
        logger.info("Training script finished or terminated.")
        # Close logger handlers if needed (e.g., WandB)
        # Check if logger has a close method or specific cleanup needed
        if hasattr(logger, 'close'):
            logger.close()
        # Explicitly finish wandb run if logger doesn't handle it
        elif config.get('logging', {}).get('use_wandb', False):
             # Check if wandb was actually imported and initialized
             if 'wandb' in sys.modules and sys.modules['wandb'] is not None and hasattr(sys.modules['wandb'], 'run') and sys.modules['wandb'].run is not None:
                  print("Finishing Weights & Biases run...") # Use print as logger might be closed
                  sys.modules['wandb'].finish()


if __name__ == "__main__":
    main()

