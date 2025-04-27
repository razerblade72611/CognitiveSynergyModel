# cognitive_synergy/models/interfaces.py
"""
Interface Modules for Cross-Modal Interaction.

Defines the BiDirectionalInterfaceModule responsible for enabling interaction
between features from corresponding vision and language backbone layers.
Includes helper modules for attention pooling, cross-attention, and FiLM gating.
Applies Gradient Checkpointing to CrossAttentionBlocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Import the checkpoint utility
import torch.utils.checkpoint as checkpoint
from typing import Dict, Tuple, Optional

# ==============================================================================
# Helper Modules
# ==============================================================================

class AttentionPooling(nn.Module):
    """ Performs attention pooling using a learnable query vector. """
    def __init__(self, dim: int, num_heads: int = 1):
        """
        Args:
            dim (int): The feature dimension of the input sequence.
            num_heads (int): Number of attention heads (often 1 for simple pooling).
        """
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Feature dimension must be positive, got {dim}")
        # Learnable query vector, initialized randomly
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        # Standard MultiheadAttention layer
        # Ensure embed_dim matches dim and is divisible by num_heads if num_heads > 1
        if dim % num_heads != 0:
             print(f"Warning: AttentionPooling dim ({dim}) not divisible by num_heads ({num_heads}). Using num_heads=1.")
             num_heads = 1
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        print(f"Initialized AttentionPooling with dim={dim}, num_heads={num_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pools the input sequence along the sequence dimension.

        Args:
            x (torch.Tensor): Input sequence tensor (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Pooled representation (batch_size, dim).
        """
        # Input validation
        if x.ndim != 3:
            raise ValueError(f"Input tensor must be 3D (batch, seq, dim), got shape {x.shape}")
        if x.shape[2] != self.query.shape[2]:
             raise ValueError(f"Input feature dimension ({x.shape[2]}) does not match "
                              f"AttentionPooling dimension ({self.query.shape[2]})")
        if x.shape[1] == 0:
             print("Warning: AttentionPooling received input with sequence length 0. Returning zeros.")
             return torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)


        batch_size = x.shape[0]
        # Expand the learnable query to match the batch size
        query = self.query.expand(batch_size, -1, -1)
        # Perform attention: query attends to the input sequence (x)
        # Output shape: (batch_size, 1, dim)
        try:
            # Use key_padding_mask if needed (e.g., if input `x` could have padding)
            # Assuming no padding mask needed for intermediate features for now
            pooled, _ = self.attn(query, x, x) # Query, Key, Value
        except Exception as e:
            print(f"Error during MultiheadAttention forward pass in AttentionPooling:")
            print(f"Query shape: {query.shape}, Key shape: {x.shape}, Value shape: {x.shape}")
            raise e
        # Remove the sequence dimension (which is 1)
        return pooled.squeeze(1) # [batch_size, dim]


class CrossAttentionBlock(nn.Module):
    """Standard Transformer-style cross-attention with residual connection and LayerNorm (Post-Norm)."""
    def __init__(self, query_dim: int, context_dim: int, hidden_dim: int, n_heads: int):
        """
        Args:
            query_dim (int): Dimension of query features.
            context_dim (int): Dimension of context features.
            hidden_dim (int): Internal dimension for attention calculation.
            n_heads (int): Number of attention heads. Must divide hidden_dim.
        """
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"CrossAttention hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        if query_dim <= 0 or context_dim <= 0 or hidden_dim <= 0 or n_heads <= 0:
             raise ValueError("Dimensions and n_heads must be positive.")

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        # Project back to the original query dimension for residual connection
        self.out_proj = nn.Linear(hidden_dim, query_dim)
        # Layer normalization layers (using Post-Norm structure here)
        self.norm_query = nn.LayerNorm(query_dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.norm_out = nn.LayerNorm(query_dim)
        print(f"Initialized CrossAttentionBlock: Q={query_dim}, C={context_dim}, H={hidden_dim}, Heads={n_heads}")


    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): Query sequence (batch, query_seq_len, query_dim).
            context (torch.Tensor): Context sequence (batch, context_seq_len, context_dim).

        Returns:
            torch.Tensor: Output sequence attended to context (batch, query_seq_len, query_dim).
        """
        # Input validation
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError(f"Query and context tensors must be 3D. Got shapes: {query.shape}, {context.shape}")
        if query.shape[0] != context.shape[0]:
             raise ValueError(f"Batch sizes must match. Got Query: {query.shape[0]}, Context: {context.shape[0]}")
        if query.shape[2] != self.norm_query.normalized_shape[0]:
             raise ValueError(f"Query dim ({query.shape[2]}) doesn't match CrossAttention query_dim ({self.norm_query.normalized_shape[0]})")
        if context.shape[2] != self.norm_context.normalized_shape[0]:
             raise ValueError(f"Context dim ({context.shape[2]}) doesn't match CrossAttention context_dim ({self.norm_context.normalized_shape[0]})")


        # Store original query for residual connection
        residual = query

        # Normalize inputs before projection (Pre-projection Norm)
        query_norm = self.norm_query(query)
        context_norm = self.norm_context(context)

        # Project query, key, value
        q = self.query_proj(query_norm)
        # Project context for key and value
        k = self.context_proj(context_norm)
        v = k  # Key and Value derived from the same context projection

        # Perform multi-head attention
        # attn_output shape: (batch, query_seq_len, hidden_dim)
        try:
            # Assuming no padding mask needed for intermediate features for now
            attn_output, _attn_weights = self.attn(q, k, v) # Query, Key, Value
        except Exception as e:
            print(f"Error during MultiheadAttention forward pass in CrossAttention:")
            print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
            raise e

        # Project back to query_dim
        projected_output = self.out_proj(attn_output)

        # Add residual connection and apply final layer normalization (Post-Norm)
        output = self.norm_out(residual + projected_output)

        return output


class FiLMGating(nn.Module):
    """FiLM modulation: dynamically scale and shift features based on conditioning."""
    def __init__(self, input_dim: int, cond_dim: int):
        """
        Args:
            input_dim (int): Dimension of the features to be modulated.
            cond_dim (int): Dimension of the conditioning vector.
        """
        super().__init__()
        if input_dim <= 0 or cond_dim <= 0:
            raise ValueError("Input and conditioning dimensions must be positive.")

        # Linear layers to predict scale (gamma) and shift (beta) parameters
        self.scale_predictor = nn.Linear(cond_dim, input_dim)
        self.shift_predictor = nn.Linear(cond_dim, input_dim)
        # Initialize predictors for identity transform initially (gamma=1, beta=0)
        # Initialize scale predictor weights/bias to zero -> predicts scale=0 -> effective gamma=1
        nn.init.zeros_(self.scale_predictor.weight)
        nn.init.zeros_(self.scale_predictor.bias)
        # Initialize shift predictor weights/bias to zero -> predicts shift=0 -> effective beta=0
        nn.init.zeros_(self.shift_predictor.weight)
        nn.init.zeros_(self.shift_predictor.bias)
        print(f"Initialized FiLMGating: InputDim={input_dim}, CondDim={cond_dim}")


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Applies FiLM modulation.

        Args:
            x (torch.Tensor): Input features to modulate (batch, seq_len, input_dim).
            cond (torch.Tensor): Conditioning vector (batch, cond_dim).

        Returns:
            torch.Tensor: Modulated features (batch, seq_len, input_dim).
        """
        # Input validation
        if x.ndim != 3:
            raise ValueError(f"Input tensor 'x' must be 3D (batch, seq, dim), got shape {x.shape}")
        if cond.ndim != 2:
             raise ValueError(f"Conditioning tensor 'cond' must be 2D (batch, dim), got shape {cond.shape}")
        if x.shape[0] != cond.shape[0]:
             raise ValueError(f"Batch sizes must match for input ({x.shape[0]}) and condition ({cond.shape[0]})")
        if x.shape[2] != self.scale_predictor.out_features:
             raise ValueError(f"Input feature dim ({x.shape[2]}) doesn't match FiLM input_dim ({self.scale_predictor.out_features})")
        if cond.shape[1] != self.scale_predictor.in_features:
             raise ValueError(f"Conditioning dim ({cond.shape[1]}) doesn't match FiLM cond_dim ({self.scale_predictor.in_features})")


        # Predict scale and shift parameters from the conditioning vector
        # Unsqueeze adds a sequence dimension for broadcasting: [batch, 1, dim]
        scale = self.scale_predictor(cond).unsqueeze(1)
        shift = self.shift_predictor(cond).unsqueeze(1)
        # Apply modulation: (1 + scale) * x + shift
        return x * (1 + scale) + shift


# ==============================================================================
# Main Bi-Directional Interface Module
# ==============================================================================

class BiDirectionalInterfaceModule(nn.Module):
    """
    Connects one layer of ViT and LLM, performs cross-modal interaction,
    pools the results, and projects to the workspace dimension.
    Applies Gradient Checkpointing to CrossAttentionBlocks.
    """
    def __init__(self, vit_dim: int, llm_dim: int, workspace_output_dim: int, config: Dict):
        """
        Args:
            vit_dim (int): Feature dimension of the ViT layer input.
            llm_dim (int): Feature dimension of the LLM layer input.
            workspace_output_dim (int): Output dimension for each modality's contribution
                                        to the shared workspace.
            config (Dict): Configuration dictionary containing parameters for sub-modules.
                           Expected keys:
                           - cross_attention_hidden_dim (int, optional)
                           - n_heads (int, optional)
                           - use_film (bool, optional, default: False)
                           - pooling_type (str, 'cls' or 'attention', default: 'attention')
                           - attention_pooling_heads (int, optional, default: 1)
        """
        super().__init__()
        self.config = config
        # Determine hidden dimension for cross-attention, defaulting sensibly
        hidden_dim = config.get('cross_attention_hidden_dim', max(vit_dim, llm_dim))
        n_heads = config.get('n_heads', 8) # Default from MVRP config
        self.use_film = config.get('use_film', False) # Default to False as decided
        self.pooling_type = config.get('pooling_type', 'attention') # Default to attention pooling
        attn_pool_heads = config.get('attention_pooling_heads', 1) # Heads for attention pooling

        if vit_dim <= 0 or llm_dim <= 0 or workspace_output_dim <= 0:
             raise ValueError("Feature dimensions must be positive.")

        print(f"Initializing BiDirectionalInterface: ViT_dim={vit_dim}, LLM_dim={llm_dim}, WS_out={workspace_output_dim}")
        print(f"  Config: hidden_dim={hidden_dim}, n_heads={n_heads}, use_film={self.use_film}, pooling={self.pooling_type}")
        print(f"  Gradient Checkpointing: ENABLED for CrossAttentionBlocks") # Added notification

        # --- Cross-Attention Blocks ---
        # LLM queries Vision context
        self.vit_to_llm_attn = CrossAttentionBlock(llm_dim, vit_dim, hidden_dim, n_heads)
        # Vision queries LLM context
        self.llm_to_vit_attn = CrossAttentionBlock(vit_dim, llm_dim, hidden_dim, n_heads)

        # --- Optional FiLM Gating ---
        if self.use_film:
            # FiLM modulates ViT features based on LLM context (pooled)
            self.vit_film = FiLMGating(vit_dim, llm_dim)
            # FiLM modulates LLM features based on ViT context (pooled)
            self.llm_film = FiLMGating(llm_dim, vit_dim)
        else:
            # Define as None if not used, simplifies forward pass check
            self.vit_film = None
            self.llm_film = None

        # --- Pooling Modules ---
        if self.pooling_type == 'attention':
            self.vit_pooler = AttentionPooling(vit_dim, attn_pool_heads)
            self.llm_pooler = AttentionPooling(llm_dim, attn_pool_heads)
        elif self.pooling_type == 'cls':
            # CLS pooling is handled by slicing in the forward pass, no module needed
            self.vit_pooler = None
            self.llm_pooler = None
        else:
            raise ValueError(f"Unsupported pooling_type: '{self.pooling_type}'. Choose 'cls' or 'attention'.")

        # --- Projection to Workspace Dimension ---
        # Project the pooled representation of each modality
        self.vit_to_workspace = nn.Linear(vit_dim, workspace_output_dim)
        self.llm_to_workspace = nn.Linear(llm_dim, workspace_output_dim)

        # Layer normalization for the final projected outputs before workspace
        self.norm_vit_out = nn.LayerNorm(workspace_output_dim)
        self.norm_llm_out = nn.LayerNorm(workspace_output_dim)


    def _pool_features(self, features: torch.Tensor, pooler: Optional[AttentionPooling], modality: str) -> torch.Tensor:
        """ Helper function to apply the configured pooling method. """
        if self.pooling_type == 'cls':
            # Assume CLS token is the first token in the sequence
            if features.shape[1] < 1: # Check if sequence length is valid
                 raise ValueError(f"Sequence length for {modality} pooling is {features.shape[1]}, cannot extract CLS token at index 0.")
            return features[:, 0, :] # [batch_size, dim]
        elif self.pooling_type == 'attention':
            if pooler is None:
                 raise RuntimeError(f"Attention pooling selected for {modality} but pooler is not initialized.")
            # Note: If AttentionPooling itself becomes a bottleneck, it could also be checkpointed.
            # For now, we focus on CrossAttention as requested.
            return pooler(features) # [batch_size, dim]
        else:
            # Should not happen due to check in __init__
            raise ValueError(f"Invalid pooling type configured: {self.pooling_type}")


    def forward(self, vit_feats: torch.Tensor, llm_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs bi-directional interaction, pooling, and projection.
        Uses Gradient Checkpointing for CrossAttentionBlocks.

        Args:
            vit_feats (torch.Tensor): Sequence features from ViT layer (batch, vit_seq_len, vit_dim).
            llm_feats (torch.Tensor): Sequence features from LLM layer (batch, llm_seq_len, llm_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Projected ViT contribution for workspace (batch, workspace_output_dim).
                - Projected LLM contribution for workspace (batch, workspace_output_dim).
        """
        # Input validation
        if vit_feats.ndim != 3 or llm_feats.ndim != 3:
             raise ValueError(f"Interface inputs must be 3D. Got ViT: {vit_feats.shape}, LLM: {llm_feats.shape}")
        if vit_feats.shape[0] != llm_feats.shape[0]:
             raise ValueError(f"Batch sizes must match for ViT ({vit_feats.shape[0]}) and LLM ({llm_feats.shape[0]})")
        # Check dimensions match expected dimensions from init (via cross-attention blocks)
        # Using query_dim from the cross-attention blocks for checking input feature dimensions
        if vit_feats.shape[2] != self.llm_to_vit_attn.norm_query.normalized_shape[0]: # llm_to_vit_attn uses vit_feats as query
             raise ValueError(f"ViT input dim ({vit_feats.shape[2]}) doesn't match expected query_dim ({self.llm_to_vit_attn.norm_query.normalized_shape[0]})")
        if llm_feats.shape[2] != self.vit_to_llm_attn.norm_query.normalized_shape[0]: # vit_to_llm_attn uses llm_feats as query
             raise ValueError(f"LLM input dim ({llm_feats.shape[2]}) doesn't match expected query_dim ({self.vit_to_llm_attn.norm_query.normalized_shape[0]})")


        # 1. Cross-Attention with Gradient Checkpointing: Modalities attend to each other
        # LLM features attended by Vision context (residual connection inside block)
        mod_llm_seq = checkpoint.checkpoint(
            self.vit_to_llm_attn,  # Function to checkpoint
            llm_feats,             # Argument 1 (query)
            vit_feats,             # Argument 2 (context)
            use_reentrant=False    # Recommended setting
        )
        # Vision features attended by LLM context (residual connection inside block)
        mod_vit_seq = checkpoint.checkpoint(
            self.llm_to_vit_attn,  # Function to checkpoint
            vit_feats,             # Argument 1 (query)
            llm_feats,             # Argument 2 (context)
            use_reentrant=False    # Recommended setting
        )

        # Features used for FiLM conditioning (use pooled versions of attended features)
        # Pool the *attended* features to get global conditioning signals
        # These poolers are initialized based on the *original* feature dimensions
        llm_cond_signal = self._pool_features(mod_llm_seq, self.llm_pooler, "LLM (for FiLM cond)")
        vit_cond_signal = self._pool_features(mod_vit_seq, self.vit_pooler, "ViT (for FiLM cond)")

        # Features to be modulated and pooled (start with the *attended* features from cross-attention)
        vit_to_pool = mod_vit_seq
        llm_to_pool = mod_llm_seq

        # 2. Optional FiLM Gating: Modulate attended features based on attended context
        if self.use_film and self.vit_film is not None and self.llm_film is not None:
            # Modulate attended ViT features using pooled attended LLM signal
            # FiLM could also be checkpointed if it becomes a memory bottleneck, e.g.:
            # vit_to_pool = checkpoint.checkpoint(self.vit_film, vit_to_pool, llm_cond_signal, use_reentrant=False)
            vit_to_pool = self.vit_film(vit_to_pool, llm_cond_signal)
            # Modulate attended LLM features using pooled attended ViT signal
            llm_to_pool = self.llm_film(llm_to_pool, vit_cond_signal)

        # 3. Pooling: Reduce sequence dimension of (potentially FiLM'd) attended features
        pooled_vit = self._pool_features(vit_to_pool, self.vit_pooler, "ViT (final)")
        pooled_llm = self._pool_features(llm_to_pool, self.llm_pooler, "LLM (final)")

        # 4. Projection to Workspace Dimension
        vit_ws_contrib = self.vit_to_workspace(pooled_vit)
        llm_ws_contrib = self.llm_to_workspace(pooled_llm)

        # Apply LayerNorm to final contributions for stability before workspace fusion
        vit_ws_contrib = self.norm_vit_out(vit_ws_contrib)
        llm_ws_contrib = self.norm_llm_out(llm_ws_contrib)

        return vit_ws_contrib, llm_ws_contrib
