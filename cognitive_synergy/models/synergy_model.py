# cognitive_synergy/models/synergy_model.py
"""
Main Cognitive Synergy Model Architecture.

This module defines the CognitiveSynergyModel class, which integrates
vision and language backbones, multi-level bi-directional interfaces,
and a shared workspace to produce fused representations. It also includes
projection heads for downstream tasks like contrastive alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# --- Assume necessary imports from other modules in the project ---
# These would be actual project imports, e.g., from .backbones import ...
# Using placeholder imports for standalone execution/review:
try:
    # Attempt to import the actual modules if they exist in the execution context
    from .backbones import ViTBackboneWrapper, BERTBackboneWrapper
    from .interfaces import BiDirectionalInterfaceModule
    from .workspace import SharedWorkspace
except ImportError:
    # Fallback to dummy classes if imports fail (e.g., running this file alone)
    print("Warning: Could not import project modules (backbones, interfaces, workspace). Using dummy classes for definition.")
    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, *args, **kwargs): return torch.randn(1) # Minimal forward
    ViTBackboneWrapper = BERTBackboneWrapper = BiDirectionalInterfaceModule = SharedWorkspace = DummyModule
    # Mock attributes needed for initialization logic
    ViTBackboneWrapper.feature_dim = 768
    BERTBackboneWrapper.feature_dim = 768


# ==============================================================================
# Projection Head Module
# ==============================================================================

class ProjectionHead(nn.Module):
    """
    Simple MLP projection head to map embeddings to a common space
    for contrastive loss calculation. Includes Layer Normalization.
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim (int): Dimension of the input embeddings.
            output_dim (int): Dimension of the projected embeddings.
        """
        super().__init__()
        # Input validation
        if input_dim <= 0 or output_dim <= 0:
             raise ValueError(f"ProjectionHead input ({input_dim}) and output ({output_dim}) dimensions must be positive.")
        # A simple 2-layer MLP: Linear -> GELU -> Linear
        # Expand intermediate dimension for potentially better representation learning
        intermediate_dim = input_dim * 2
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.activation = nn.GELU() # GELU is common in transformer-based models
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        # Normalize the final projected output for stability in contrastive loss
        self.layer_norm = nn.LayerNorm(output_dim)
        print(f"Initialized ProjectionHead: InputDim={input_dim}, OutputDim={output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the input tensor.

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_dim).

        Returns:
            torch.Tensor: Projected tensor (batch_size, output_dim).
        """
        # Input validation
        if x.ndim != 2:
             raise ValueError(f"ProjectionHead input tensor must be 2D (batch, dim), got shape {x.shape}")
        if x.shape[1] != self.fc1.in_features:
             raise ValueError(f"Input tensor dim ({x.shape[1]}) does not match projection head input dim ({self.fc1.in_features})")

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x


# ==============================================================================
# Main Cognitive Synergy Model
# ==============================================================================

class CognitiveSynergyModel(nn.Module):
    """
    Main model orchestrating ViT and LLM interaction via multi-level
    bi-directional interfaces and a shared workspace. Includes projection
    heads for contrastive alignment loss.
    """
    def __init__(self, config: Dict):
        """
        Initializes the Cognitive Synergy Model based on the provided configuration dictionary.

        Args:
            config (Dict): Configuration dictionary matching the structure expected
                           (e.g., loaded from 'mvrp_v1.yaml'). Requires keys like
                           'backbones', 'interface_layers', 'interface_module',
                           'shared_workspace', 'contrastive_head'.
        """
        super().__init__()
        print("Initializing CognitiveSynergyModel...")
        self.config = config

        # --- Validate essential config sections ---
        required_sections = ['backbones', 'interface_layers', 'interface_module', 'shared_workspace', 'contrastive_head']
        for section in required_sections:
            if section not in config:
                raise KeyError(f"Configuration missing required section: '{section}'")
            if not isinstance(config[section], dict):
                 # Interface layers is list, handle that case
                 if section == 'interface_layers' and not isinstance(config[section], list):
                     # Correction: interface_layers itself should be a dict containing lists
                     if not isinstance(config[section], dict):
                          raise TypeError(f"Config section '{section}' must be a dictionary, got {type(config[section])}")
                 elif section != 'interface_layers':
                      raise TypeError(f"Config section '{section}' must be a dictionary, got {type(config[section])}")


        # --- 1. Instantiate Backbone Wrappers ---
        vision_config = config['backbones'].get('vision', {})
        language_config = config['backbones'].get('language', {})
        interface_layers_config = config['interface_layers']

        # Validate nested configs
        if 'model_name' not in vision_config: raise KeyError("Missing 'model_name' in backbones.vision config.")
        if 'model_name' not in language_config: raise KeyError("Missing 'model_name' in backbones.language config.")
        if 'vision' not in interface_layers_config or not isinstance(interface_layers_config['vision'], list):
             raise KeyError("Missing or invalid 'vision' list in 'interface_layers' config.")
        if 'language' not in interface_layers_config or not isinstance(interface_layers_config['language'], list):
             raise KeyError("Missing or invalid 'language' list in 'interface_layers' config.")

        self.vision_backbone = ViTBackboneWrapper(
            model_name=vision_config['model_name'],
            feature_layers=interface_layers_config['vision'],
            pretrained=vision_config.get('pretrained', True) # Default to pretrained=True
        )
        self.language_backbone = BERTBackboneWrapper(
            model_name=language_config['model_name'],
            feature_layers=interface_layers_config['language'],
            pretrained=language_config.get('pretrained', True) # Default to pretrained=True
        )

        # Store backbone feature dimensions
        self.vision_feature_dim = self.vision_backbone.feature_dim
        self.language_feature_dim = self.language_backbone.feature_dim

        # Basic dimension check (informational)
        if self.vision_feature_dim != self.language_feature_dim:
            print(f"Info: Vision backbone dim ({self.vision_feature_dim}) != "
                  f"Language backbone dim ({self.language_feature_dim}). Interfaces handle this.")

        # --- 2. Instantiate Bi-Directional Interface Modules ---
        if len(interface_layers_config['vision']) != len(interface_layers_config['language']):
            raise ValueError("Number of vision and language interface layers must match.")

        self.interfaces = nn.ModuleList()
        num_interfaces = len(interface_layers_config['vision'])
        if num_interfaces == 0:
             print("Warning: No interface layers specified. The model will run backbones but perform no fusion.")

        interface_module_config = config['interface_module']
        workspace_config = config['shared_workspace'] # Corrected key name

        # Get the dimension of contributions projected by the interface module
        workspace_input_dim_per_iface = interface_module_config.get('workspace_output_dim', 256) # Match default in config

        print(f"Creating {num_interfaces} BiDirectionalInterfaceModule(s)...")
        for i in range(num_interfaces):
            interface = BiDirectionalInterfaceModule(
                vit_dim=self.vision_feature_dim,
                llm_dim=self.language_feature_dim,
                workspace_output_dim=workspace_input_dim_per_iface,
                config=interface_module_config # Pass the whole interface config dict
            )
            self.interfaces.append(interface)

        # --- 3. Instantiate Shared Workspace ---
        # Check if workspace config keys exist
        required_ws_keys = ['num_layers', 'hidden_dim', 'output_dim']
        for key in required_ws_keys:
             if key not in workspace_config:
                 raise KeyError(f"Missing required key '{key}' in shared_workspace config.")

        self.workspace = SharedWorkspace(
            num_interfaces=num_interfaces,
            interface_contribution_dim=workspace_input_dim_per_iface,
            num_layers=workspace_config['num_layers'],
            hidden_dim=workspace_config['hidden_dim'],
            output_dim=workspace_config['output_dim'], # Final world state dimension
            n_heads=interface_module_config.get('n_heads', 8) # Use same n_heads as interfaces by default
        )
        self.world_state_dim = workspace_config['output_dim']
        print(f"Created SharedWorkspace with output dimension: {self.world_state_dim}")

        # --- 4. Instantiate Contrastive Projection Head ---
        contrastive_head_config = config['contrastive_head']
        self.use_projection_head = contrastive_head_config.get('use_projection', True)

        if self.use_projection_head:
            if 'projection_dim' not in contrastive_head_config:
                 raise KeyError("Missing 'projection_dim' in 'contrastive_head' config when use_projection is true.")
            projection_dim = contrastive_head_config['projection_dim']
            print(f"Creating ProjectionHeads with output dimension: {projection_dim}")

            # Projector for the final world_state output
            self.world_state_proj_head = ProjectionHead(
                input_dim=self.world_state_dim,
                output_dim=projection_dim
            )
            # Projector for the vision CLS token (from backbone)
            self.vision_cls_proj_head = ProjectionHead(
                input_dim=self.vision_feature_dim,
                output_dim=projection_dim
            )
            # Projector for the language CLS token (from backbone)
            self.language_cls_proj_head = ProjectionHead(
                input_dim=self.language_feature_dim,
                output_dim=projection_dim
            )
        else:
             # If not using projection, use Identity layers (no-op)
             print("ProjectionHead disabled. Using nn.Identity(). Ensure downstream loss handles raw dimensions.")
             self.world_state_proj_head = nn.Identity()
             self.vision_cls_proj_head = nn.Identity()
             self.language_cls_proj_head = nn.Identity()
             # Add a check if dimensions happen to match (unlikely but possible)
             if self.world_state_dim != self.vision_feature_dim or self.world_state_dim != self.language_feature_dim:
                 print(f"  Warning: Raw dimensions may differ (WS:{self.world_state_dim}, V:{self.vision_feature_dim}, L:{self.language_feature_dim}). "
                       "Contrastive loss might fail without projection if dimensions aren't manually aligned.")

        print("CognitiveSynergyModel initialization complete.")


    def forward(self,
                image_input: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the main forward pass of the Cognitive Synergy Model.

        Args:
            image_input (torch.Tensor): Input image tensor for ViT (B, C, H, W).
            input_ids (torch.Tensor): Input token IDs for BERT (B, SeqLen).
            attention_mask (torch.Tensor): Attention mask for BERT (B, SeqLen).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing model outputs, including:
                - 'world_state': Raw output from the SharedWorkspace (B, world_state_dim).
                - 'projected_world_state': World state projected for contrastive loss (B, proj_dim or world_state_dim).
                - 'projected_vision_cls': Vision CLS token projected for contrastive loss (B, proj_dim or vision_feature_dim).
                - 'projected_language_cls': Language CLS token projected for contrastive loss (B, proj_dim or language_feature_dim).
        """

        # --- 1. Run Backbones ---
        # Get final CLS token and dictionary of intermediate layer features
        vision_cls_token, vision_intermediates = self.vision_backbone(image_input)
        language_cls_token, language_intermediates = self.language_backbone(input_ids, attention_mask)

        # --- 2. Process through Interfaces ---
        workspace_contributions = []
        vision_layers = self.config['interface_layers']['vision']
        language_layers = self.config['interface_layers']['language']

        if len(self.interfaces) != len(vision_layers):
             # This should not happen if init logic is correct, but good safeguard
             raise RuntimeError(f"Mismatch between number of initialized interfaces ({len(self.interfaces)}) "
                                f"and specified layers ({len(vision_layers)}).")

        for i in range(len(self.interfaces)):
            # Get features for the current interface level
            vit_layer_idx = vision_layers[i]
            llm_layer_idx = language_layers[i]

            # Retrieve features, ensuring the keys exist
            if vit_layer_idx not in vision_intermediates:
                 raise KeyError(f"Vision layer index {vit_layer_idx} not found in intermediate features. Available keys: {list(vision_intermediates.keys())}")
            if llm_layer_idx not in language_intermediates:
                 raise KeyError(f"Language layer index {llm_layer_idx} not found in intermediate features. Available keys: {list(language_intermediates.keys())}")

            vit_feat = vision_intermediates[vit_layer_idx]
            llm_feat = language_intermediates[llm_layer_idx]

            # Pass features through the corresponding interface module
            # Interface module handles interaction, pooling, and projection to contribution dim
            ws_contribution_vit, ws_contribution_llm = self.interfaces[i](vit_feat, llm_feat)

            # Collect contributions for the workspace
            workspace_contributions.append((ws_contribution_vit, ws_contribution_llm))

        # --- 3. Fuse in Shared Workspace ---
        # The workspace module takes the list of (vit_contrib, llm_contrib) tuples
        # Handle case where no interfaces were defined (num_interfaces=0)
        if self.interfaces:
            world_state = self.workspace(workspace_contributions) # Shape: [B, world_state_dim]
        else:
             # If no interfaces, world_state cannot be computed. Return zeros or handle differently?
             # For now, return zeros with the expected dimension.
             batch_size = vision_cls_token.shape[0]
             world_state = torch.zeros(batch_size, self.world_state_dim, device=vision_cls_token.device, dtype=vision_cls_token.dtype)
             print("Warning: No interfaces defined, returning zero tensor for world_state.")


        # --- 4. Project Outputs for Contrastive Loss (or Identity if disabled) ---
        projected_world_state = self.world_state_proj_head(world_state)
        projected_vision_cls = self.vision_cls_proj_head(vision_cls_token)
        projected_language_cls = self.language_cls_proj_head(language_cls_token)

        # --- 5. Prepare Output Dictionary ---
        # This dictionary structure allows flexibility in the training loop
        outputs = {
            "world_state": world_state, # Raw world state output, potentially useful for other tasks/losses
            "projected_world_state": projected_world_state, # Projected for contrastive loss
            "projected_vision_cls": projected_vision_cls,   # Projected for contrastive loss
            "projected_language_cls": projected_language_cls, # Projected for contrastive loss
        }

        return outputs


