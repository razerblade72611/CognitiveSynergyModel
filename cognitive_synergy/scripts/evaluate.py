# cognitive_synergy/scripts/evaluate.py
"""
Evaluation Script for the Cognitive Synergy Model.

This script loads a trained model checkpoint and evaluates its performance
on a specified dataset (e.g., validation or test set). Calculates loss
and optionally other metrics like retrieval recall.
"""

import argparse
import yaml # PyYAML for loading config: pip install pyyaml
import os
import sys # For exiting on critical errors
import pprint # For pretty printing config
import torch
import torch.nn as nn # <--- ADDED THIS IMPORT
import torch.nn.functional as F # For normalization if needed
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from tqdm import tqdm # Progress bar
import time # For timing evaluation
import logging # Use standard logging

# --- Import project components ---
# Use try-except blocks for robustness if run before all modules exist
try:
    # Assuming the script is run from the project root (e.g., My_Last_Project/Something_new/)
    # and the cognitive_synergy package is in the Python path
    from cognitive_synergy.models import CognitiveSynergyModel
    from cognitive_synergy.data.datasets import ContrastiveImageTextDataset # Or a factory function
    from cognitive_synergy.training.losses import get_loss_function # <-- ADD THIS LINE
    from cognitive_synergy.data.transforms import get_image_transform, TextTransform
    from cognitive_synergy.data.dataloaders import create_dataloader
    from cognitive_synergy.training.losses import ContrastiveAlignmentLoss # Or get_loss_function
    from cognitive_synergy.utils.logging import setup_logger
    from cognitive_synergy.utils.metrics import calculate_retrieval_recall_at_k # Example metric
    # Import other necessary metric functions as needed
    # from cognitive_synergy.utils.metrics import calculate_vqa_accuracy, get_predicted_answers_from_logits
except ImportError as e:
    print(f"Error importing project modules in scripts/evaluate.py: {e}")
    print("Please ensure all required modules exist and the script is run such that the 'cognitive_synergy' package is importable.")
    # Define dummy classes/functions to allow script structure definition if needed for linting
    class Dummy: pass
    def dummy_func(*args, **kwargs): pass
    CognitiveSynergyModel = ContrastiveImageTextDataset = Dummy
    get_image_transform = TextTransform = create_dataloader = ContrastiveAlignmentLoss = setup_logger = calculate_retrieval_recall_at_k = dummy_func
    # Exit if imports fail in a real scenario
    sys.exit(f"Failed to import necessary modules: {e}")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate the Cognitive Synergy Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file used during training (e.g., effective_config.yaml)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file (.pth.tar) to evaluate."
    )
    parser.add_argument(
        "--eval_manifest",
        type=str,
        required=True,
        help="Path to the manifest file for the evaluation dataset (e.g., test_manifest.json)."
    )
    # --- Optional Overrides ---
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None, # Default to value in config
        help="Override evaluation batch size from config."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None, # Default to value in config or auto-detect
        help="Override device (e.g., 'cuda:0', 'cpu')."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override number of data loader workers."
    )
    # Add arguments for specific evaluation tasks or metrics if needed
    parser.add_argument(
        "--calculate_recall",
        action="store_true",
        help="Calculate retrieval recall metrics (requires contrastive setup)."
    )
    parser.add_argument(
        "--recall_k",
        type=int,
        nargs='+',
        default=[1, 5, 10],
        help="K values for Recall@K calculation."
    )
    parser.add_argument(
        "--no_loss",
        action="store_true",
        help="Do not calculate loss during evaluation."
    )

    args = parser.parse_args()
    return args

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config is None:
             raise ValueError(f"Configuration file {config_path} is empty or invalid.")
        print(f"Loaded configuration from: {config_path}")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        print(f"Error loading configuration file {config_path}: {e}")
        raise

@torch.no_grad() # Disable gradient calculations for evaluation efficiency
def evaluate(
    model: nn.Module, # Now nn is defined
    dataloader: DataLoader,
    criterion: Optional[nn.Module], # Now nn is defined
    device: torch.device,
    logger: logging.Logger, # Use specific type hint
    config: Dict, # Pass config for potential metric settings
    calculate_recall: bool = False,
    recall_k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Runs the evaluation loop on the provided dataloader.

    Args:
        model (nn.Module): The loaded model (already on the correct device).
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        criterion (Optional[nn.Module]): Loss function (e.g., contrastive loss) to calculate validation loss.
        device (torch.device): The device to run evaluation on.
        logger (logging.Logger): Logger instance for logging progress and results.
        config (Dict): Configuration dictionary for potential metric settings.
        calculate_recall (bool): Flag to enable retrieval recall calculation.
        recall_k_values (List[int]): K values for recall calculation.

    Returns:
        Dict[str, float]: Dictionary containing calculated evaluation metrics.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    num_samples = 0
    start_time = time.time()

    # --- Store embeddings for potential retrieval calculation ---
    # Only store if needed for specific metrics (can consume significant memory)
    all_vision_embeddings = [] if calculate_recall else None
    all_language_embeddings = [] if calculate_recall else None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating", leave=False)

    for step, batch in pbar:
        # --- Move data to device ---
        try:
            # Reuse the _prepare_batch logic if available or implement here
            prepared_batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            image_input = prepared_batch['image']
            input_ids = prepared_batch['input_ids']
            attention_mask = prepared_batch['attention_mask']
            current_batch_size = image_input.shape[0]
            num_samples += current_batch_size
        except KeyError as e:
            logger.error(f"Missing key {e} in evaluation batch at step {step}. Skipping batch.", exc_info=True)
            continue
        except Exception as e:
            logger.error(f"Error moving evaluation batch to device at step {step}: {e}. Skipping batch.", exc_info=True)
            continue

        # --- Forward Pass ---
        try:
            model_outputs = model(image_input, input_ids, attention_mask)
            # Use .get() for safer access to potentially missing keys
            vision_emb = model_outputs.get('projected_vision_cls')
            language_emb = model_outputs.get('projected_language_cls')

            if vision_emb is None or language_emb is None:
                 logger.error("Missing projected embeddings ('projected_vision_cls' or 'projected_language_cls') "
                              "in model output during evaluation. Skipping batch.")
                 continue

            # Calculate loss if criterion is provided
            batch_loss = None
            if criterion:
                loss = criterion(vision_emb, language_emb)
                total_loss += loss.item() * current_batch_size # Accumulate loss weighted by batch size
                batch_loss = loss.item()

            # Store embeddings if calculating retrieval metrics
            if calculate_recall and all_vision_embeddings is not None and all_language_embeddings is not None:
                 # Move to CPU to save GPU memory during accumulation
                 all_vision_embeddings.append(vision_emb.cpu())
                 all_language_embeddings.append(language_emb.cpu())

            # --- Calculate other batch-level metrics here if needed ---
            # Example: ITM accuracy (would need ITM head output and labels)
            # if 'itm_logits' in model_outputs and 'itm_labels' in prepared_batch:
            #     itm_logits = model_outputs['itm_logits']
            #     itm_targets = prepared_batch['itm_labels']
            #     # ... calculate batch ITM accuracy ...

        except Exception as e:
             logger.error(f"Error during evaluation forward/loss calculation at step {step}: {e}. Skipping batch.", exc_info=True)
             continue

        pbar.set_postfix(loss=f"{batch_loss:.4f}" if batch_loss is not None else "N/A")

    # --- Calculate Final Metrics ---
    eval_metrics = {}
    if criterion and num_samples > 0:
        avg_loss = total_loss / num_samples
        eval_metrics['eval_loss'] = avg_loss
        logger.info(f"Evaluation Average Loss: {avg_loss:.5f}")
    elif criterion:
         logger.warning("Criterion provided but no samples processed or loss calculation failed. Loss not reported.")
         eval_metrics['eval_loss'] = float('nan')

    # Calculate retrieval metrics if requested and embeddings were collected
    if calculate_recall and all_vision_embeddings and all_language_embeddings:
        logger.info(f"Calculating retrieval metrics for K={recall_k_values}...")
        try:
            # Concatenate all embeddings from the list of batch tensors
            vision_emb_all = torch.cat(all_vision_embeddings, dim=0)
            lang_emb_all = torch.cat(all_language_embeddings, dim=0)
            logger.info(f"  Total embeddings concatenated for retrieval: {vision_emb_all.shape[0]}")

            if vision_emb_all.shape[0] != lang_emb_all.shape[0]:
                 logger.error(f"  Mismatch in number of collected vision ({vision_emb_all.shape[0]}) and "
                              f"language ({lang_emb_all.shape[0]}) embeddings. Skipping recall calculation.")
            elif vision_emb_all.shape[0] == 0:
                 logger.warning("  No embeddings collected. Skipping recall calculation.")
            else:
                # Ensure embeddings are normalized (might already be from projection head)
                vision_emb_all = F.normalize(vision_emb_all.float(), p=2, dim=-1) # Ensure float type
                lang_emb_all = F.normalize(lang_emb_all.float(), p=2, dim=-1)

                # Calculate similarity matrix (potentially large, consider chunking if OOM occurs)
                # Assuming evaluation fits in memory for now. Use CPU for large matrices.
                logger.info("  Calculating similarity matrix...")
                sim_matrix = torch.matmul(vision_emb_all, lang_emb_all.t())

                # Calculate Recall@K using the utility function
                recall_results = calculate_retrieval_recall_at_k(sim_matrix.cpu(), k_values=recall_k_values) # Calculate on CPU
                eval_metrics.update(recall_results) # Add R@K results to the main metrics dict
                logger.info(f"  Retrieval Results: {recall_results}")

        except Exception as e:
             logger.error(f"Error calculating retrieval metrics: {e}", exc_info=True)

    # Add other aggregated metric calculations here
    # Example: Average ITM accuracy over all batches
    # if total_samples_itm > 0:
    #     avg_itm_accuracy = total_correct_itm / total_samples_itm
    #     eval_metrics['itm_accuracy'] = avg_itm_accuracy
    #     logger.info(f"Average ITM Accuracy: {avg_itm_accuracy:.4f}")


    eval_time = time.time() - start_time
    eval_metrics['eval_time_sec'] = eval_time
    eval_metrics['num_samples'] = num_samples
    logger.info(f"Evaluation finished in {eval_time:.2f} seconds for {num_samples} samples.")

    return eval_metrics


def main():
    """Main function to setup and run evaluation."""
    args = parse_args()
    config = load_config(args.config)
    # Apply CLI overrides to config *after* loading (optional, but useful)
    # config = override_config_with_args(config, args) # Need this function if using overrides

    # --- Setup ---
    # Pretty print the configuration used for training (loaded from training run)
    print("\n--- Training Configuration ---")
    pprint.pprint(config, indent=2)
    print("--------------------------\n")

    # Setup Logger - use a dedicated log file for this evaluation run
    eval_log_dir = os.path.join(os.path.dirname(args.checkpoint), "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_log_file = os.path.join(eval_log_dir, f"eval_{os.path.splitext(os.path.basename(args.checkpoint))[0]}_{os.path.splitext(os.path.basename(args.eval_manifest))[0]}.log")
    # Disable WandB for evaluation script by default, can be enabled via config if needed
    logger = setup_logger(config=config, log_file=eval_log_file, use_wandb=False)
    logger.info("Starting evaluation script...")
    logger.info(f"Using training configuration from: {args.config}")
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Evaluation dataset manifest: {args.eval_manifest}")

    # Setup Device
    if args.device:
        device_str = args.device
    else:
        # Use device from config, default to auto-detect
        device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    if device_str.startswith('cuda') and not torch.cuda.is_available():
        logger.warning(f"CUDA device '{device_str}' requested but CUDA not available. Switching to CPU.")
        device = torch.device('cpu')
    else:
        try:
             device = torch.device(device_str)
             _ = torch.tensor([1]).to(device) # Test device availability
        except Exception as e:
             logger.error(f"Could not set device to '{device_str}'. Error: {e}. Falling back to CPU.", exc_info=True)
             device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Setting up evaluation data loader...")
    data_config = config.get('data', {})
    # Use batch size override if provided, else use validation batch size from config, else train batch size
    eval_batch_size = args.batch_size if args.batch_size is not None \
        else config.get('validation', {}).get('batch_size', config.get('training', {}).get('batch_size', 64))
    num_workers = args.num_workers if args.num_workers is not None \
        else config.get('training', {}).get('num_workers', 0) # Reuse num_workers setting

    # Create transforms (using validation settings)
    try:
        text_transform = TextTransform(
            tokenizer_name=config.get('backbones', {}).get('language', {}).get('model_name', 'bert-base-uncased'),
            max_length=data_config.get('max_text_length', 128)
        )
        eval_image_transform = get_image_transform(
            image_size=data_config.get('image_size', 224),
            is_train=False # Use validation transforms for evaluation
        )
    except Exception as e:
         logger.error(f"Error creating data transforms: {e}", exc_info=True)
         sys.exit(1)

    # Create evaluation dataset
    # Assuming Contrastive dataset for eval based on MVRP focus. Adapt if evaluating VQA etc.
    try:
        eval_dataset = ContrastiveImageTextDataset(
            manifest_path=args.eval_manifest,
            image_transform=eval_image_transform,
            text_transform=text_transform,
            image_root=data_config.get('image_root', None)
        )
    except Exception as e:
         logger.error(f"Error creating evaluation dataset from {args.eval_manifest}: {e}", exc_info=True)
         sys.exit(1)

    # Create evaluation dataloader
    try:
        eval_loader = create_dataloader(
            dataset=eval_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=False, # No shuffling for evaluation
            pin_memory=config.get('training', {}).get('pin_memory', True),
            drop_last=False, # Do not drop last batch for evaluation
        )
    except Exception as e:
         logger.error(f"Error creating evaluation dataloader: {e}", exc_info=True)
         sys.exit(1)
    logger.info(f"Evaluation DataLoader: {len(eval_loader)} batches, Batch Size: {eval_batch_size}")

    # --- Model Initialization and Loading Checkpoint ---
    logger.info("Initializing model architecture...")
    try:
        # Initialize model using the *training* config to ensure architecture matches
        model = CognitiveSynergyModel(config=config).to(device)
    except Exception as e:
         logger.error(f"Error initializing CognitiveSynergyModel: {e}", exc_info=True)
         sys.exit(1)

    logger.info(f"Loading model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle potential DDP prefix 'module.' in state dict keys
        state_dict = checkpoint['model_state_dict']
        if all(k.startswith('module.') for k in state_dict.keys()):
            logger.info("Detected 'module.' prefix in checkpoint state_dict, removing it.")
            state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}
        # Load weights, allow partial loading if needed (strict=False)
        load_result = model.load_state_dict(state_dict, strict=True)
        logger.info(f"Model load result: {load_result}") # Check for missing/unexpected keys
        logger.info(f"Successfully loaded model weights trained up to epoch {checkpoint.get('epoch', 'N/A')}")
    except Exception as e:
         logger.error(f"Error loading model state_dict from checkpoint: {e}", exc_info=True)
         sys.exit(1)

    # --- Loss Function (Optional for Eval) ---
    criterion = None
    # Check CLI flag first, then config
    calculate_loss_flag = not args.no_loss if args.no_loss is not None \
        else config.get('evaluation', {}).get('calculate_loss', True)

    if calculate_loss_flag:
        logger.info("Initializing loss function for evaluation...")
        try:
            criterion = get_loss_function(config).to(device) # Use factory based on training config
        except Exception as e:
             logger.warning(f"Error initializing loss function: {e}. Loss will not be calculated.", exc_info=True)
             criterion = None
    else:
         logger.info("Loss calculation disabled for evaluation.")


    # --- Run Evaluation ---
    logger.info("Starting evaluation...")
    try:
        eval_results = evaluate(
            model=model,
            dataloader=eval_loader,
            criterion=criterion,
            device=device,
            logger=logger,
            config=config, # Pass config for metric settings
            calculate_recall=args.calculate_recall,
            recall_k_values=args.recall_k
        )
    except Exception as e:
         logger.error("An error occurred during evaluation.", exc_info=True)
         sys.exit(1)

    # --- Report Results ---
    logger.info("\n--- Evaluation Results ---")
    # Sort results for consistent output
    sorted_results = dict(sorted(eval_results.items()))
    for metric, value in sorted_results.items():
        # Format floats nicely
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.5f}")
        else:
            logger.info(f"  {metric}: {value}")
    logger.info("------------------------")

    # Optionally save results to a file
    results_save_path = os.path.join(os.path.dirname(args.checkpoint), f"eval_results_{os.path.splitext(os.path.basename(args.checkpoint))[0]}_{os.path.splitext(os.path.basename(args.eval_manifest))[0]}.yaml")
    try:
        # Convert tensors in results to float/list if any exist before saving
        saveable_results = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in sorted_results.items()}
        with open(results_save_path, 'w') as f:
            yaml.dump(saveable_results, f, default_flow_style=False, indent=2)
        logger.info(f"Evaluation results saved to: {results_save_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")

    logger.info("Evaluation script finished.")

if __name__ == "__main__":
    main()

