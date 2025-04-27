# cognitive_synergy/training/__init__.py
"""
Initialization file for the training package.

This file makes the 'training' directory a Python package, allowing
modules within it (like losses.py, trainer.py, optimizers.py) to be imported
using standard Python import syntax (e.g., from cognitive_synergy.training.trainer import Trainer).
"""

# This file can be left empty or can be used to expose specific classes/functions
# from modules within the 'training' package for convenience, improving import paths.

# Example (optional): Expose key classes/functions directly
# try:
#     from .losses import ContrastiveAlignmentLoss, get_loss_function # Assuming defined in losses.py
#     from .trainer import Trainer # Assuming defined in trainer.py
#     from .optimizers import create_optimizer, create_scheduler # Assuming defined in optimizers.py
#     # from .curriculum import CurriculumManager # Assuming defined in curriculum.py
# except ImportError as e:
#      print(f"Warning: Could not perform optional imports in training/__init__.py: {e}")
#      # Define dummy classes if needed for static analysis when submodules don't exist yet
#      class DummyLoss: pass
#      class DummyTrainer: pass
#      def create_optimizer(*args, **kwargs): pass
#      def create_scheduler(*args, **kwargs): pass
#      # class DummyCurriculum: pass
#      ContrastiveAlignmentLoss = get_loss_function = DummyLoss
#      Trainer = DummyTrainer
#      # CurriculumManager = DummyCurriculum


# Define __all__ to control `from .training import *` behavior (optional)
# __all__ = [
#     "ContrastiveAlignmentLoss",
#     "get_loss_function",
#     "Trainer",
#     "create_optimizer",
#     "create_scheduler",
#     # "CurriculumManager",
# ] # Example

print("cognitive_synergy.training package initialized.")


