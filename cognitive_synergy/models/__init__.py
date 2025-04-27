# cognitive_synergy/models/__init__.py
"""
Initialization file for the models package.

This file makes the 'models' directory a Python package and exports
key classes for easier access from other parts of the project.
"""

# Import classes from backbones.py
try:
    from .backbones import ViTBackboneWrapper, BERTBackboneWrapper
except ImportError as e:
    print(f"Warning: Could not import from .backbones in models/__init__.py: {e}")
    # Define dummy classes if needed for partial execution/linting
    class DummyBackbone: pass
    ViTBackboneWrapper = BERTBackboneWrapper = DummyBackbone

# Import classes from interfaces.py
try:
    from .interfaces import (
        BiDirectionalInterfaceModule,
        CrossAttentionBlock,
        FiLMGating,
        AttentionPooling
    )
except ImportError as e:
    print(f"Warning: Could not import from .interfaces in models/__init__.py: {e}")
    class DummyInterface: pass
    BiDirectionalInterfaceModule = CrossAttentionBlock = FiLMGating = AttentionPooling = DummyInterface

# Import class from workspace.py
try:
    from .workspace import SharedWorkspace
except ImportError as e:
    print(f"Warning: Could not import from .workspace in models/__init__.py: {e}")
    class DummyWorkspace: pass
    SharedWorkspace = DummyWorkspace

# Import classes from synergy_model.py
try:
    from .synergy_model import CognitiveSynergyModel, ProjectionHead
except ImportError as e:
    print(f"Warning: Could not import from .synergy_model in models/__init__.py: {e}")
    class DummySynergy: pass
    CognitiveSynergyModel = ProjectionHead = DummySynergy

# Import classes from prediction_heads.py
try:
    from .prediction_heads import (
        ImageTextMatchingHead,
        VQAPredictionHead,          # Placeholder included
        CausalLMPredictionHead      # Placeholder included
    )
except ImportError as e:
    print(f"Warning: Could not import from .prediction_heads in models/__init__.py: {e}")
    class DummyHead: pass
    ImageTextMatchingHead = VQAPredictionHead = CausalLMPredictionHead = DummyHead


# Define __all__ to control what `from .models import *` imports (optional but good practice)
# This helps manage the namespace and clarifies the public API of the package.
__all__ = [
    # Backbones
    "ViTBackboneWrapper",
    "BERTBackboneWrapper",
    # Interfaces
    "BiDirectionalInterfaceModule",
    "CrossAttentionBlock",
    "FiLMGating",
    "AttentionPooling",
    # Workspace
    "SharedWorkspace",
    # Main Model & Projection
    "CognitiveSynergyModel",
    "ProjectionHead",
    # Prediction Heads
    "ImageTextMatchingHead",
    "VQAPredictionHead",
    "CausalLMPredictionHead",
]

print("cognitive_synergy.models package initialized.")


