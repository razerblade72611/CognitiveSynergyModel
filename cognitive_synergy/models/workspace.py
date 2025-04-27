# cognitive_synergy/models/workspace.py
"""
Shared Workspace Module.

Defines the SharedWorkspace class which fuses contributions from multiple
interface levels using a Transformer Encoder to produce a unified 'world_state'.
Includes Gradient Checkpointing for Transformer layers.
"""

import torch
import torch.nn as nn
# Import the checkpoint utility
import torch.utils.checkpoint as checkpoint
from typing import List, Tuple, Dict

class SharedWorkspace(nn.Module):
    """
    Fuses contributions from multiple bi-directional interface levels
    into a unified 'world_state' representation using a Transformer Encoder.
    Uses Gradient Checkpointing on Transformer layers for memory efficiency.
    """
    def __init__(self,
                 num_interfaces: int,
                 interface_contribution_dim: int,
                 num_layers: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_heads: int = 8): # Added n_heads for TransformerEncoderLayer consistency
        """
        Args:
            num_interfaces (int): The number of interface levels providing input.
                                  This determines the sequence length for the Transformer.
            interface_contribution_dim (int): The feature dimension of the contribution
                                                from *each* modality (ViT or LLM) per interface.
            num_layers (int): Number of layers in the Transformer Encoder.
            hidden_dim (int): The internal dimension of the Transformer Encoder layers (d_model).
            output_dim (int): The final dimension of the output 'world_state'.
            n_heads (int): Number of attention heads in the Transformer Encoder layers. Must divide hidden_dim.
        """
        super().__init__()
        # Input validation
        if num_interfaces <= 0:
            raise ValueError(f"Number of interfaces must be positive, got {num_interfaces}")
        if interface_contribution_dim <= 0:
             raise ValueError(f"Interface contribution dimension must be positive, got {interface_contribution_dim}")
        if num_layers <= 0:
             raise ValueError(f"Number of workspace layers must be positive, got {num_layers}")
        if hidden_dim <= 0:
             raise ValueError(f"Workspace hidden dimension must be positive, got {hidden_dim}")
        if output_dim <= 0:
             raise ValueError(f"Workspace output dimension must be positive, got {output_dim}")
        if n_heads <= 0:
             raise ValueError(f"Number of attention heads must be positive, got {n_heads}")
        if hidden_dim % n_heads != 0:
            raise ValueError(f"Workspace hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")

        self.num_interfaces = num_interfaces
        self.interface_contribution_dim = interface_contribution_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Calculate the dimension after concatenating ViT and LLM contributions per interface level
        self.concatenated_input_dim = 2 * interface_contribution_dim
        print(f"Initializing SharedWorkspace: NumInterfaces={num_interfaces}, ContribDim={interface_contribution_dim}")
        print(f"  Config: NumLayers={num_layers}, HiddenDim={hidden_dim}, OutputDim={output_dim}, Heads={n_heads}")
        print(f"  Gradient Checkpointing: ENABLED for Transformer Encoder layers") # Added notification

        # --- Input Projection ---
        # Project the concatenated input from each interface level to the Transformer's hidden dimension
        self.input_projection = nn.Linear(self.concatenated_input_dim, hidden_dim)

        # --- Transformer Encoder ---
        # Standard Transformer Encoder Layer configuration
        # Note: We still create the standard nn.TransformerEncoder, but we will call its layers
        # individually using checkpointing in the forward pass.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4, # Standard practice: feedforward dim is 4*d_model
            dropout=0.1, # Standard dropout rate, consider making configurable
            activation='relu', # Standard activation, could be configurable (e.g., 'gelu')
            batch_first=True # Expect input as (batch, seq, feature)
        )
        # Stack the encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # --- Output Projection ---
        # Project the pooled output of the Transformer Encoder to the final world_state dimension
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Layer normalization for stability at input and output stages
        self.layer_norm_in = nn.LayerNorm(hidden_dim)
        self.layer_norm_out = nn.LayerNorm(output_dim)


    def forward(self, interface_outputs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Processes the list of contributions from interfaces and outputs the world_state.
        Uses Gradient Checkpointing for the Transformer Encoder layers.

        Args:
            interface_outputs (List[Tuple[torch.Tensor, torch.Tensor]]):
                A list where each element is a tuple (vit_contribution, llm_contribution).
                Each contribution tensor has shape [batch_size, interface_contribution_dim].
                The length of the list must equal self.num_interfaces.

        Returns:
            torch.Tensor: The final world_state tensor of shape [batch_size, output_dim].
        """
        # --- Input Validation ---
        if not isinstance(interface_outputs, list):
             raise TypeError(f"Expected interface_outputs as a list, got {type(interface_outputs)}")
        if len(interface_outputs) != self.num_interfaces:
            raise ValueError(f"Expected {self.num_interfaces} interface outputs, but received {len(interface_outputs)}")
        if not interface_outputs: # Handle empty list case
             raise ValueError("Received empty list for interface_outputs.")

        # Check tensor shapes and batch sizes within the list
        expected_shape_part = (self.interface_contribution_dim,)
        batch_size = interface_outputs[0][0].shape[0] # Get batch size from first tensor
        for i, (vit_contrib, llm_contrib) in enumerate(interface_outputs):
             if not isinstance(vit_contrib, torch.Tensor) or not isinstance(llm_contrib, torch.Tensor):
                 raise TypeError(f"Contributions at index {i} must be tensors, got {type(vit_contrib)}, {type(llm_contrib)}")
             if vit_contrib.ndim != 2 or llm_contrib.ndim != 2:
                 raise ValueError(f"Contribution tensors must be 2D (batch, dim). Got shapes {vit_contrib.shape}, {llm_contrib.shape} at index {i}")
             if vit_contrib.shape[1:] != expected_shape_part or llm_contrib.shape[1:] != expected_shape_part:
                 raise ValueError(f"Contribution tensors at index {i} have incorrect dimension. Expected (*, {self.interface_contribution_dim}), "
                                  f"got ViT: {vit_contrib.shape}, LLM: {llm_contrib.shape}")
             if vit_contrib.shape[0] != batch_size or llm_contrib.shape[0] != batch_size:
                 raise ValueError(f"Batch sizes inconsistent across interface contributions at index {i}. Expected {batch_size}, got {vit_contrib.shape[0]}, {llm_contrib.shape[0]}.")


        device = interface_outputs[0][0].device # Get device from first tensor

        # --- Prepare Input Sequence for Transformer ---
        concatenated_contributions = []
        for vit_contrib, llm_contrib in interface_outputs:
            # Concatenate ViT and LLM contributions for this interface level
            # Shape: [batch_size, 2 * interface_contribution_dim]
            concat_pair = torch.cat((vit_contrib, llm_contrib), dim=-1)
            concatenated_contributions.append(concat_pair)

        # Stack the concatenated pairs along the sequence dimension (dim=1)
        # This creates the sequence input for the Transformer Encoder.
        # Shape: [batch_size, num_interfaces, 2 * interface_contribution_dim]
        transformer_input_sequence = torch.stack(concatenated_contributions, dim=1)

        # --- Project Input Sequence ---
        # Project to the Transformer's hidden dimension
        # Shape: [batch_size, num_interfaces, hidden_dim]
        projected_input = self.input_projection(transformer_input_sequence)
        # Apply LayerNorm after projection
        projected_input = self.layer_norm_in(projected_input)

        # --- Pass through Transformer Encoder with Gradient Checkpointing ---
        # Instead of: transformer_output = self.transformer_encoder(projected_input)
        # We loop through the layers and apply checkpointing to each one.
        hidden_state = projected_input
        # Access the layers within the nn.TransformerEncoder module
        for layer_module in self.transformer_encoder.layers:
            # Apply checkpointing to each layer
            hidden_state = checkpoint.checkpoint(
                layer_module,          # The module to run
                hidden_state,          # Input tensor (src)
                None,                  # src_mask (assuming not used based on original call)
                None,                  # src_key_padding_mask (assuming not used)
                use_reentrant=False    # Recommended for newer PyTorch versions
                # If using older PyTorch or specific TransformerEncoderLayer versions,
                # you might need use_reentrant=True or adjust arguments if masks are needed.
            )
        transformer_output = hidden_state # Output of the last layer
        # Shape: [batch_size, num_interfaces, hidden_dim]


        # --- Pool Transformer Output ---
        # Average pooling along the sequence dimension (representing interface levels)
        # This aggregates information across all interface interactions.
        # Shape: [batch_size, hidden_dim]
        pooled_output = transformer_output.mean(dim=1)

        # --- Project to Final Output Dimension ---
        # Shape: [batch_size, output_dim]
        world_state = self.output_projection(pooled_output)
        # Apply final LayerNorm for stability
        world_state = self.layer_norm_out(world_state)

        return world_state
