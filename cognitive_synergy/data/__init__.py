# cognitive_synergy/data/__init__.py
"""
Initialization file for the data package.

This file makes the 'data' directory a Python package, allowing
modules within it (like datasets.py, transforms.py) to be imported
using standard Python import syntax.
"""

# This file can be left empty or can be used to expose specific classes/functions
# from modules within the 'data' package for convenience.

# Example (optional): Expose key classes directly from submodules
# try:
#     from .datasets import ContrastiveImageTextDataset, VQADataset # Assuming defined in datasets.py
#     from .transforms import get_image_transform, TextTransform # Assuming defined in transforms.py
#     from .dataloaders import create_dataloader # Assuming defined in dataloaders.py
# except ImportError as e:
#      print(f"Warning: Could not perform optional imports in data/__init__.py: {e}")
#      # Define dummy classes if needed for static analysis when submodules don't exist yet
#      class DummyDataset: pass
#      class DummyTransform: pass
#      def create_dataloader(*args, **kwargs): pass
#      ContrastiveImageTextDataset = VQADataset = DummyDataset
#      get_image_transform = TextTransform = DummyTransform


# Define __all__ to control `from .data import *` behavior (optional)
# __all__ = [
#     "ContrastiveImageTextDataset",
#     "VQADataset",
#     "get_image_transform",
#     "TextTransform",
#     "create_dataloader",
# ] # Example

print("cognitive_synergy.data package initialized.")


