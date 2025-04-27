# cognitive_synergy/utils/__init__.py
"""
Initialization file for the utils package.

This file makes the 'utils' directory a Python package, allowing
modules within it (like logging.py, metrics.py, misc.py) to be imported
using standard Python import syntax (e.g., from cognitive_synergy.utils.logging import setup_logger).
"""

# This file can be left empty or can be used to expose specific classes/functions
# from modules within the 'utils' package for convenience, improving import paths.

# Example (optional): Expose key functions/classes directly
try:
    from .logging import setup_logger # Assuming defined in logging.py
    from .metrics import calculate_accuracy, calculate_retrieval_recall_at_k, calculate_vqa_accuracy # Assuming defined in metrics.py
    from .misc import set_seed, save_config # Assuming defined in misc.py
except ImportError as e:
     # This might happen if utils submodules haven't been created yet or have issues
     print(f"Warning: Could not perform optional imports in utils/__init__.py: {e}")
     # Define dummy classes/functions if needed for static analysis or partial runs
     def setup_logger(*args, **kwargs): pass
     def calculate_accuracy(*args, **kwargs): pass
     def calculate_retrieval_recall_at_k(*args, **kwargs): pass
     def calculate_vqa_accuracy(*args, **kwargs): pass
     def set_seed(*args, **kwargs): pass
     def save_config(*args, **kwargs): pass


# Define __all__ to control `from cognitive_synergy.utils import *` behavior (optional)
# List the names of objects intended to be the public API of this package.
__all__ = [
    "setup_logger",
    "calculate_accuracy",
    "calculate_retrieval_recall_at_k",
    "calculate_vqa_accuracy",
    "set_seed",
    "save_config",
] # Example based on the optional imports above

print("cognitive_synergy.utils package initialized.")

