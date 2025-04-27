# cognitive_synergy/utils/misc.py
"""
Miscellaneous Utility Functions.

Contains helper functions for tasks like setting random seeds for reproducibility,
potentially saving configurations, or other general utilities needed across the project.
"""

import random
import numpy as np
import torch
import os
import yaml # Using PyYAML for config saving example: pip install pyyaml
from typing import Dict, Any, Optional
import logging # Use standard logging

# Get a logger instance
logger = logging.getLogger("CognitiveSynergy.Misc") # Use hierarchical naming

# ==============================================================================
# Reproducibility
# ==============================================================================

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across relevant libraries (random, numpy, torch).

    Args:
        seed (int): The seed value to use. Must be an integer.
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed)}")

    # Set seed for Python's random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch CPU operations
    torch.manual_seed(seed)

    # Set seed for PyTorch CUDA operations (if CUDA is available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Set seed for all GPUs (if using multi-GPU)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"Set seed to {seed} for random, numpy, torch, and cuda.")

        # Optional: Configure deterministic algorithms for CUDA.
        # Note: This can potentially slow down training and might not cover all operations.
        # It's often used for debugging or strict reproducibility checks.
        # Consider making this configurable via the main config file.
        # try:
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False # Benchmark must be False for deterministic
        #     logger.info("  Enabled CUDA deterministic algorithms (may impact performance).")
        # except AttributeError:
        #     logger.warning("  Could not set torch.backends.cudnn properties (might be older PyTorch version).")

    else:
        logger.info(f"Set seed to {seed} for random, numpy, and torch (CUDA not available).")


# ==============================================================================
# Configuration Handling (Example)
# ==============================================================================

def save_config(config: Dict[str, Any], filepath: str):
    """
    Saves the configuration dictionary to a YAML file.

    Args:
        config (Dict[str, Any]): The configuration dictionary to save.
        filepath (str): The path to the output YAML file (e.g., 'output/run_config.yaml').
    """
    # Input validation
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dictionary, got {type(config)}")
    if not filepath:
         raise ValueError("Filepath for saving config cannot be empty.")
    # Check for standard YAML extensions
    if not filepath.endswith(('.yaml', '.yml')):
        logger.warning(f"Saving config to file without standard .yaml/.yml extension: {filepath}")

    try:
        # Ensure the directory exists before attempting to write the file
        dir_name = os.path.dirname(filepath)
        if dir_name: # Only create if path includes a directory component
             # exist_ok=True prevents error if directory already exists
             os.makedirs(dir_name, exist_ok=True)

        # Save using PyYAML, preserving order if possible (sort_keys=False) and using indentation
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
        logger.info(f"Configuration saved successfully to {filepath}")

    except Exception as e:
        logger.error(f"Error saving configuration to {filepath}: {e}", exc_info=True)
        # Optionally re-raise the exception if saving the config is critical
        # raise e

# Example Usage (when running this file directly)
if __name__ == "__main__":
    # Setup basic logging for testing this module
    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s | %(name)s | %(message)s')

    print("--- Testing Misc Utilities ---")

    # Seed Setting
    print("\n1. Setting Seed")
    set_seed(123)
    r1, n1, t1 = random.random(), np.random.rand(), torch.rand(1).item()
    logger.info(f"  Seed 123: random={r1:.4f}, numpy={n1:.4f}, torch={t1:.4f}")
    set_seed(123) # Reset seed
    r2, n2, t2 = random.random(), np.random.rand(), torch.rand(1).item()
    logger.info(f"  Reset 123: random={r2:.4f}, numpy={n2:.4f}, torch={t2:.4f}")
    assert r1 == r2 and n1 == n2 and t1 == t2, "Seed setting failed!"
    logger.info("  Seed reset verified.")

    # Config Saving
    print("\n2. Saving Config")
    dummy_config = {
        'project': 'CognitiveSynergy_Test',
        'model': {'name': 'test_model', 'layers': 4, 'interface_settings': None},
        'training': {'lr': 1e-3, 'batch_size': 16, 'epochs': 10},
        'data': {'path': '/data/set', 'augment': True},
        'list_example': [1, 2, 3, {'nested': True}]
    }
    save_dir = "temp_config_test_dir" # Use a subdirectory
    save_path = os.path.join(save_dir, "temp_test_config.yaml")

    # Clean up previous test file/dir if it exists
    if os.path.exists(save_path): os.remove(save_path)
    if os.path.exists(save_dir) and not os.listdir(save_dir): os.rmdir(save_dir)

    save_config(dummy_config, save_path)

    # Verify save (optional)
    if os.path.exists(save_path):
        logger.info(f"  Config file '{save_path}' created.")
        # Optionally load and check content
        try:
            with open(save_path, 'r') as f_read:
                loaded_config = yaml.safe_load(f_read)
            assert loaded_config == dummy_config, "Loaded config doesn't match saved config!"
            logger.info("  Config content verified.")
        except Exception as e:
            logger.error(f"  Error verifying config content: {e}")

        # Clean up
        try:
            os.remove(save_path)
            if os.path.exists(save_dir) and not os.listdir(save_dir): os.rmdir(save_dir)
            logger.info(f"  Cleaned up '{save_path}' and potentially '{save_dir}'.")
        except OSError as e:
             logger.error(f"  Error during cleanup: {e}")
    else:
        logger.error(f"  Error: Config file '{save_path}' was not created.")
