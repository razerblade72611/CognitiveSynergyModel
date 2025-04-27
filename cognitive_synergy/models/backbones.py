# cognitive_synergy/models/backbones.py
"""
Backbone Wrappers for Vision (ViT) and Language (BERT) models.

These wrappers provide a consistent interface to load pre-trained models
and extract intermediate layer features along with the final CLS token embedding.
Uses Gradient Checkpointing for internal Transformer blocks/layers.

ViT Wrapper now uses Hugging Face transformers library.
BERT Wrapper uses Hugging Face transformers library.
"""

import torch
import torch.nn as nn
# Removed timm import
# import timm
# Added transformers imports for ViT
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoImageProcessor, DebertaV2Tokenizer
from typing import Dict, List, Union, Tuple
# Import the checkpoint utility
import torch.utils.checkpoint as checkpoint

# ==============================================================================
# Vision Backbone Wrapper (Now using Hugging Face transformers)
# ==============================================================================

class ViTBackboneWrapper(nn.Module):
    """
    Wrapper for Hugging Face ViT models (like DINOv2) to extract intermediate layer features.
    Uses Gradient Checkpointing on Transformer layers.
    Provides a consistent interface for feature extraction.
    """
    def __init__(self, model_name: str = 'facebook/dinov2-small', feature_layers: List[int] = [2, 6, 10], pretrained: bool = True):
        """
        Args:
            model_name (str): Name of the ViT-like model on Hugging Face Hub (e.g., 'facebook/dinov2-small').
            feature_layers (List[int]): List of layer indices (0-based) to extract features from.
                                        Layer 0 is embedding output. Layers 1 to N are transformer layers.
            pretrained (bool): Whether to load pretrained weights (ignored if model loaded manually).
                                Transformers AutoModel handles pretraining implicitly.
        """
        super().__init__()
        self.model_name = model_name
        # Ensure sorted unique layers for consistent ordering
        self.feature_layers = sorted(list(set(feature_layers)))

        print(f"Initializing ViT Wrapper for HF Model: {model_name}")
        print(f"  Gradient Checkpointing: ENABLED for Transformer Layers") # Added notification

        # Load config, ensuring hidden states are output
        # Add trust_remote_code=True if the model requires custom code execution
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True) # trust_remote_code=True)
        # Load the model itself using the config
        # Add trust_remote_code=True if needed
        self.model = AutoModel.from_pretrained(model_name, config=config) # , trust_remote_code=True)
        # Load an image processor (optional, but good practice if needed later or for consistency)
        # self.processor = AutoImageProcessor.from_pretrained(model_name)
        print(f"Loaded HF ViT model {model_name} successfully.")

        # --- Store Feature Dimension ---
        self.feature_dim = self.model.config.hidden_size
        print(f"ViT feature dimension set to: {self.feature_dim}")
        print(f"ViT intermediate features will be extracted from layers: {self.feature_layers}")

        # --- Validate Layers ---
        self._validate_layers()

        # --- Check for HF gradient checkpointing config ---
        if hasattr(self.model.config, 'gradient_checkpointing') and self.model.config.gradient_checkpointing:
            print(f"Warning: Hugging Face model config for {model_name} has 'gradient_checkpointing=True'. "
                  "Explicit checkpointing in this wrapper might be redundant.")


    def _validate_layers(self):
        """Checks if the requested feature layer indices are valid for HF ViT models."""
        # Total layers available = embedding layer (idx 0) + transformer layers (idx 1 to N)
        # HF ViT models typically store num layers in config.num_hidden_layers
        if not hasattr(self.model.config, 'num_hidden_layers'):
             raise AttributeError(f"Model {self.model_name} config missing 'num_hidden_layers'. Cannot validate feature layers.")
        num_encoder_layers = self.model.config.num_hidden_layers
        num_layers_available = num_encoder_layers + 1 # Embeddings (0) + Layers (1..N)

        if not self.feature_layers:
            print("Warning: No feature layers requested for ViT.")
            return
        max_layer_req = max(self.feature_layers)
        # Check if requested layers are within the valid range
        if max_layer_req >= num_layers_available:
             raise ValueError(f"Requested layer index {max_layer_req} is out of bounds for {self.model_name}. "
                              f"Available indices: 0 (embeddings) to {num_encoder_layers} (final layer). "
                              f"Total available: {num_layers_available}.")
        if min(self.feature_layers) < 0:
             raise ValueError("Feature layer indices must be non-negative.")


    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Performs a forward pass using gradient checkpointing on the encoder layers,
        returning the final CLS token embedding and intermediate features.

        Args:
            pixel_values (torch.Tensor): Input image tensor (batch_size, channels, height, width).
                                         Name changed from 'x' to match HF ViT input naming.

        Returns:
            Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
                - CLS token embedding from the *last* layer (batch_size, feature_dim).
                - Dictionary mapping layer indices to intermediate feature tensors
                  (batch_size, sequence_length, feature_dim).
        """
        intermediate_features: Dict[int, torch.Tensor] = {}

        # --- Explicit HF ViT Forward Pass Logic ---

        # 1. Embeddings (Patch + Position + potentially CLS token)
        if not hasattr(self.model, 'embeddings'):
             raise AttributeError(f"Model {self.model_name} is missing standard 'embeddings'. Cannot run explicit forward pass.")
        # The specific call might vary slightly based on ViT implementation, but typically:
        embedding_output = self.model.embeddings(pixel_values)

        # Capture embedding output if requested (layer 0)
        if 0 in self.feature_layers:
            intermediate_features[0] = embedding_output.clone()

        # 2. Transformer Encoder Layers with Checkpointing
        # Assumes standard HF structure: model.encoder.layer
        if not hasattr(self.model, 'encoder') or not hasattr(self.model.encoder, 'layer'):
             # DINOv2 models might just have model.layer or model.blocks
             if hasattr(self.model, 'layer') and isinstance(self.model.layer, nn.ModuleList):
                 encoder_layers = self.model.layer
             elif hasattr(self.model, 'blocks') and isinstance(self.model.blocks, nn.ModuleList): # Common ViT structure
                 encoder_layers = self.model.blocks
             else:
                 raise AttributeError(f"Model {self.model_name} is missing standard 'encoder.layer', 'layer', or 'blocks'. Cannot run explicit forward pass.")
        else:
             encoder_layers = self.model.encoder.layer


        hidden_state = embedding_output
        layer_idx_offset = 1 # ViT layers are typically indexed 1..N

        # Prepare head mask (usually None if not used)
        head_mask = None # Or self.model.get_head_mask(None, self.model.config.num_hidden_layers) if needed

        for i, layer_module in enumerate(encoder_layers):
            current_layer_idx = i + layer_idx_offset

            # Define arguments for the specific layer's forward method
            # HF ViT Layer signature often: forward(hidden_states, head_mask, ...)
            # Check specific model if needed, but usually just hidden_states is required input
            layer_args = (hidden_state, head_mask) # Pass head_mask if layer expects it

            # Apply checkpointing
            # Layer output is often a tuple: (hidden_state, optional_attention_probs)
            layer_outputs = checkpoint.checkpoint(
                layer_module,   # Module to run
                hidden_state,   # First argument to layer_module.forward
                head_mask,      # Second argument (pass None if not needed by layer)
                # Add other args if layer_module.forward needs them
                use_reentrant=True # Recommended setting
            )

            # Extract the hidden state from the layer output (usually the first element)
            if isinstance(layer_outputs, tuple):
                hidden_state = layer_outputs[0]
            else: # Should not happen for standard HF ViT layers
                hidden_state = layer_outputs

            # Capture the output if this is a requested layer
            if current_layer_idx in self.feature_layers:
                intermediate_features[current_layer_idx] = hidden_state.clone()

        # 3. Final Layer Normalization (Optional - sometimes applied before CLS pool)
        # Check if the model has a final layernorm after the encoder
        if hasattr(self.model, 'layernorm') and isinstance(self.model.layernorm, nn.LayerNorm):
             final_hidden_state = self.model.layernorm(hidden_state)
        else:
             # Some models (like DINOv2) might apply norm within blocks or not have a final one before pooling
             final_hidden_state = hidden_state


        # --- Extract CLS Token ---
        # CLS token from the final layer's output sequence, typically at index 0
        if final_hidden_state.ndim != 3 or final_hidden_state.shape[1] == 0:
             raise ValueError(f"Final hidden state has unexpected shape {final_hidden_state.shape}. Cannot extract CLS token.")
        cls_token = final_hidden_state[:, 0, :].clone()

        # --- Final Check for Captured Features ---
        captured_layers = set(intermediate_features.keys())
        requested_layers = set(self.feature_layers)
        if captured_layers != requested_layers:
            missing = requested_layers - captured_layers
            print(f"Warning: ViT Feature capture mismatch (likely during checkpointing). "
                  f"Requested: {requested_layers}, Captured: {captured_layers}. Missing: {missing}")

        return cls_token, intermediate_features


# ==============================================================================
# Language Backbone Wrapper (BERT using Hugging Face transformers)
# ==============================================================================

class BERTBackboneWrapper(nn.Module):
    """
    Wrapper for Hugging Face BERT models to extract intermediate layer features.
    Uses Gradient Checkpointing on Transformer layers.
    Provides a consistent interface for feature extraction.
    """
    def __init__(self, model_name: str = 'bert-base-uncased', feature_layers: List[int] = [1, 6, 11], pretrained: bool = True):
        """
        Args:
            model_name (str): Name of the BERT-like model on Hugging Face Hub.
            feature_layers (List[int]): List of layer indices (0-based) to extract features from.
                                        Layer 0 is embedding output. Layers 1 to N are transformer layers.
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()
        self.model_name = model_name
        # Ensure sorted unique layers for consistent ordering
        self.feature_layers = sorted(list(set(feature_layers)))

        print(f"Initializing BERT Wrapper for {model_name} with pretrained={pretrained}")
        print(f"  Gradient Checkpointing: ENABLED for Transformer Layers") # Added notification

        # Load config - output_hidden_states not strictly needed now but doesn't hurt
        # Add trust_remote_code=True if the model requires custom code execution
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True) # , trust_remote_code=True)
        # Load the model itself
        # Add trust_remote_code=True if needed
        self.model = AutoModel.from_pretrained(model_name, config=config) if pretrained else AutoModel.from_config(config) # , trust_remote_code=True)
        
        print("--- Loaded LLM Structure ---") # Add this
        print(self.model)                   # Add this
        print("--------------------------") # Add this
        
        # Load the tokenizer associated with the model
        # Add trust_remote_code=True if needed
        self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-small') #, trust_remote_code=True)
        print(f"Loaded {model_name} model and tokenizer successfully.")

        # --- Store Feature Dimension ---
        self.feature_dim = self.model.config.hidden_size
        print(f"BERT feature dimension set to: {self.feature_dim}")
        print(f"BERT intermediate features will be extracted from layers: {self.feature_layers}")

        # --- Validate Layers ---
        self._validate_layers()

        # --- Check if gradient checkpointing is enabled in the underlying model ---
        if hasattr(self.model.config, 'gradient_checkpointing') and self.model.config.gradient_checkpointing:
            print(f"Warning: Hugging Face model config for {model_name} has 'gradient_checkpointing=True'. "
                  "Explicit checkpointing in this wrapper might be redundant.")
        # We can optionally force the underlying model's setting if needed:
        # self.model.gradient_checkpointing_disable() # To ensure only our wrapper's checkpointing is active
        # self.model.gradient_checkpointing_enable() # To use only the HF internal one (remove wrapper logic then)


    def _validate_layers(self):
        """Checks if the requested feature layer indices are valid."""
        # HF models typically store num layers in config.num_hidden_layers
        if not hasattr(self.model.config, 'num_hidden_layers'):
             raise AttributeError(f"Model {self.model_name} config missing 'num_hidden_layers'. Cannot validate feature layers.")
        num_encoder_layers = self.model.config.num_hidden_layers
        num_layers_available = num_encoder_layers + 1 # Embeddings (0) + Layers (1..N)

        if not self.feature_layers:
            print("Warning: No feature layers requested for BERT.")
            return
        max_layer_req = max(self.feature_layers)
        if max_layer_req >= num_layers_available:
             raise ValueError(f"Requested layer index {max_layer_req} is out of bounds for {self.model_name}. "
                              f"Available indices: 0 to {num_encoder_layers}. Total available: {num_layers_available}.")
        if min(self.feature_layers) < 0:
             raise ValueError("Feature layer indices must be non-negative.")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Performs a forward pass using the standard Hugging Face model call
        and extracts intermediate features based on the output_hidden_states flag.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
                - CLS token embedding from the *last* layer (batch_size, feature_dim).
                - Dictionary mapping layer indices to intermediate feature tensors
                  (batch_size, sequence_length, feature_dim).
        """
        # Perform the forward pass using the standard model call.
        # output_hidden_states=True was set during model loading in __init__.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # No need to pass position_ids explicitly here, model handles it internally
            output_hidden_states=True, # Explicitly ensure again
            return_dict=True
        )

        # hidden_states is a tuple: (embedding_output, layer1_output, ..., layerN_output)
        hidden_states = outputs.get('hidden_states', None)

        if not isinstance(hidden_states, tuple) or not hidden_states:
            raise RuntimeError(f"Failed to get 'hidden_states' tuple from {self.model_name}. Output keys: {outputs.keys()}")

        # --- Extract CLS Token ---
        # CLS token embedding from the *last* layer's output.
        # Taking the first token state as pseudo-CLS for models like Qwen2.
        last_layer_hidden_state = hidden_states[-1]
        if last_layer_hidden_state.ndim != 3 or last_layer_hidden_state.shape[1] == 0:
             raise ValueError(f"Last layer hidden state has unexpected shape {last_layer_hidden_state.shape}. Cannot extract first token.")
        cls_token = last_layer_hidden_state[:, 0, :].clone() # [batch_size, hidden_dim]

        # --- Extract Intermediate Features ---
        intermediate_features: Dict[int, torch.Tensor] = {}
        for layer_idx in self.feature_layers:
            if layer_idx < len(hidden_states):
                # Clone the tensor to prevent potential downstream modification issues
                intermediate_features[layer_idx] = hidden_states[layer_idx].clone()
            else:
                 # Safeguard
                 print(f"Warning: Layer index {layer_idx} requested but model only has {len(hidden_states)-1} layers available in output tuple.")

        # --- Final Check for Captured Features ---
        captured_layers = set(intermediate_features.keys())
        requested_layers = set(self.feature_layers)
        if captured_layers != requested_layers:
             missing = requested_layers - captured_layers
             print(f"Warning: {self.model_name} Feature capture mismatch. "
                   f"Requested: {requested_layers}, Captured: {captured_layers}. Missing: {missing}")

        return cls_token, intermediate_features
