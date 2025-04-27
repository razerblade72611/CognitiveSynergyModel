# cognitive_synergy/utils/logging.py
"""
Logging setup for the project.

Configures logging to console and optionally to file and Weights & Biases.
"""

import logging
import os
import sys
from typing import Dict, Optional, Any, Union

# Optional: Try importing wandb
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None # Assign None if wandb is not installed
    _WANDB_AVAILABLE = False

# Define a default logger name
DEFAULT_LOGGER_NAME = "CognitiveSynergy"

def setup_logger(
    config: Optional[Dict] = None,
    log_level: Optional[Union[int, str]] = None, # Allow level override
    log_file: Optional[str] = None,
    use_wandb: Optional[bool] = None,
    wandb_config: Optional[Dict] = None
) -> logging.Logger:
    """
    Sets up the main logger for the application.

    Configures logging to console, optionally to a file, and optionally
    initializes Weights & Biases logging if requested and available.
    Settings are primarily read from the 'logging' section of the config dictionary,
    but can be overridden by direct arguments.

    Args:
        config (Optional[Dict]): Full configuration dictionary. Used to extract
                                 logging and WandB settings if specific args aren't provided.
        log_level (Optional[Union[int, str]]): Override for the logging level
                                               (e.g., logging.INFO, "DEBUG").
        log_file (Optional[str]): Override for the path to the file to save logs.
                                  If None, file logging is disabled unless specified in config.
                                  If explicitly "", disables file logging regardless of config.
        use_wandb (Optional[bool]): Override for enabling Weights & Biases logging.
        wandb_config (Optional[Dict]): Override for WandB initialization configuration
                                       (project, entity, run_name, etc.).

    Returns:
        logging.Logger: The configured logger instance with an added 'log_metrics' method.
    """
    # --- Determine Configuration ---
    if config is None:
        config = {}

    # Extract relevant configuration sections safely using .get()
    logging_config = config.get('logging', {})
    # train_config = config.get('training', {}) # Used for run name fallback if needed

    # Determine log level: Argument > Config > Default (INFO)
    level_setting = log_level if log_level is not None else logging_config.get('level', 'INFO')
    if isinstance(level_setting, str):
        level = getattr(logging, level_setting.upper(), logging.INFO)
    elif isinstance(level_setting, int):
        # Ensure it's a valid logging level constant
        level = level_setting if level_setting in logging._levelToName else logging.INFO
    else:
        print(f"Warning: Invalid log_level type ({type(level_setting)}). Defaulting to INFO.")
        level = logging.INFO

    # Determine log file path: Argument > Config > Default (None)
    # Explicitly passing "" disables file logging
    if log_file == "":
        log_file_path = None
    elif log_file is not None:
        log_file_path = log_file
    else:
        # Get from config, default to None if not present
        log_file_path = logging_config.get('log_file', None)
        # Ensure None if the value is an empty string in config
        if log_file_path == "":
             log_file_path = None

    # Determine WandB usage: Argument > Config > Default (False)
    if use_wandb is None:
        use_wandb_flag = logging_config.get('use_wandb', False)
    else:
        use_wandb_flag = use_wandb

    # Determine WandB config: Argument > Config['logging']['wandb'] > Default ({})
    if wandb_config is not None:
         wandb_settings = wandb_config
    else:
        # Get wandb section from logging config, default to empty dict
        wandb_settings = logging_config.get('wandb', {})


    # --- Setup Basic Logger ---
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    logger.setLevel(level)
    # Prevent messages from propagating to the root logger if handlers are added here
    logger.propagate = False

    # Remove existing handlers to avoid duplicate logs if setup_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Configure Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    # Define a standard format for logs
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level) # Console shows messages at the specified level or higher
    logger.addHandler(console_handler)

    # --- Configure File Handler (Optional) ---
    if log_file_path:
        try:
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                 os.makedirs(log_dir, exist_ok=True)
                 logger.info(f"Created log directory: {log_dir}")

            # Create file handler (append mode)
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            # Use a more detailed format for file logs
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(level) # File logs messages at the specified level or higher
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file_path}")
        except Exception as e:
            # Log error to console if file logging setup fails
            logger.error(f"Failed to configure file logging to {log_file_path}: {e}", exc_info=True)
    else:
        logger.info("File logging disabled.")

    # --- Configure Weights & Biases (Optional) ---
    wandb_run = None # Store wandb run object if initialized
    if use_wandb_flag:
        if _WANDB_AVAILABLE:
            try:
                # Extract WandB settings, providing defaults or falling back
                project = wandb_settings.get('wandb_project', config.get('project_name', 'cognitive_synergy'))
                entity = wandb_settings.get('wandb_entity', None) # Often required, user must set this
                run_name = wandb_settings.get('wandb_run_name', None) # Optional run name

                if not entity:
                     # WandB often requires an entity (username or team name)
                     logger.warning("WandB entity not specified in config (logging.wandb.wandb_entity). "
                                    "WandB initialization might fail or use defaults found in environment/settings.")

                # Prepare arguments for wandb.init
                wandb_init_kwargs = {
                    'project': project,
                    'name': run_name, # wandb handles None name by auto-generating
                    'config': config, # Log the entire configuration dictionary for reproducibility
                    'reinit': True, # Allow reinitialization if called multiple times (e.g., in sweeps)
                    # 'mode': "online" # Use "disabled" or "offline" for debugging without cloud sync
                }
                # Only pass entity if it's explicitly provided and not None/empty
                if entity:
                    wandb_init_kwargs['entity'] = entity

                # Initialize WandB run
                wandb_run = wandb.init(**wandb_init_kwargs)

                logger.info(f"Weights & Biases logging enabled and initialized.")
                logger.info(f"  Project: {project}")
                if entity: logger.info(f"  Entity: {entity}")
                logger.info(f"  Run Name: {wandb_run.name if wandb_run else 'N/A'}")
                logger.info(f"  Run ID: {wandb_run.id if wandb_run else 'N/A'}")

            except Exception as e:
                logger.error(f"Failed to initialize Weights & Biases: {e}", exc_info=True)
                # Disable flag if initialization fails to prevent errors in log_metrics
                use_wandb_flag = False
        else:
            logger.warning("Weights & Biases logging requested but 'wandb' library is not installed. Skipping.")
            use_wandb_flag = False # Ensure flag is False if wandb not available
    else:
        logger.info("Weights & Biases logging disabled.")


    # --- Add a helper method for logging metrics ---
    # This allows calling logger.log_metrics(...) which logs via standard logging AND wandb if enabled
    def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, commit: Optional[bool] = None):
        """Logs metrics to standard logger and optionally to WandB."""
        if not isinstance(metrics, dict):
             logger.warning(f"log_metrics expected a dictionary, but received {type(metrics)}. Skipping.")
             return

        # Log to standard logger (console/file) using INFO level
        # Format floats nicely
        log_items = []
        for k, v in metrics.items():
             if isinstance(v, float):
                 log_items.append(f"{k}: {v:.5f}") # More precision for metrics
             else:
                 log_items.append(f"{k}: {v}")
        log_str = " | ".join(log_items)

        step_prefix = f"Step: {step:>7} | " if step is not None else ""
        logger.info(f"{step_prefix}METRICS | {log_str}")

        # Log to WandB if enabled and initialized
        # Use the captured wandb_run object state (checks if wandb.init was successful)
        if use_wandb_flag and wandb_run is not None:
            try:
                # Default commit=True unless specified otherwise by the caller
                # This ensures metrics are logged together for a given step by default.
                commit_flag = True if commit is None else commit
                wandb.log(metrics, step=step, commit=commit_flag)
            except Exception as e:
                # Log error but don't crash the training
                logger.error(f"Failed to log metrics to WandB: {e}", exc_info=True)

    # Attach the helper method to the logger instance for convenient access
    # Use setattr to avoid potential type conflicts if logger already has this attribute from elsewhere
    setattr(logger, 'log_metrics', log_metrics)

    logger.info(f"Logger setup complete. Effective level: {logging.getLevelName(logger.level)}")
    return logger


# Example Usage (when running this file directly)
if __name__ == "__main__":
    print("--- Testing Logger Setup ---")

    # Example 1: Basic console logging
    print("\n1. Basic Console Logging (INFO level)")
    logger1 = setup_logger(log_level=logging.INFO)
    logger1.debug("This debug message should NOT appear.")
    logger1.info("This info message should appear.")
    logger1.warning("This is a warning message.")
    logger1.log_metrics({"val_loss": 0.5678, "epoch": 1}, step=1000)

    # Example 2: Console and File logging (DEBUG level)
    print("\n2. Console and File Logging (DEBUG level)")
    log_file_path = "test_log.log"
    if os.path.exists(log_file_path):
        try: os.remove(log_file_path) # Clean up previous test log
        except OSError as e: print(f"Could not remove old log file {log_file_path}: {e}")
    logger2 = setup_logger(log_level="DEBUG", log_file=log_file_path) # Use string level name
    logger2.debug("This debug message should appear in console and file.")
    logger2.info("This info message should also appear.")
    logger2.log_metrics({"train_acc": 0.85}, step=50)
    print(f"Check '{log_file_path}' for file output (if permissions allow).")

    # Example 3: Using Config Dictionary (WandB disabled by default)
    print("\n3. Logging via Config Dictionary")
    test_config = {
        'logging': {
            'level': 'INFO',
            'log_file': 'config_test_log.log',
            'use_wandb': False, # Set to True to test wandb if installed and logged in
            'wandb': {
                'wandb_project': 'cognitive_synergy_test',
                'wandb_entity': None # Set your entity here if testing wandb
            }
        },
        'project_name': 'test_project_from_config'
    }
    log_file_path_config = test_config['logging']['log_file']
    if os.path.exists(log_file_path_config):
        try: os.remove(log_file_path_config)
        except OSError as e: print(f"Could not remove old log file {log_file_path_config}: {e}")
    logger3 = setup_logger(config=test_config)
    logger3.debug("Debug message from config setup (should not appear).")
    logger3.info("Info message from config setup.")
    logger3.log_metrics({"accuracy": 0.95, "loss": 0.123456}, step=200)
    print(f"Check '{log_file_path_config}' for file output (if permissions allow).")

    # Clean up log files (optional)
    # if os.path.exists(log_file_path): os.remove(log_file_path)
    # if os.path.exists(log_file_path_config): os.remove(log_file_path_config)


