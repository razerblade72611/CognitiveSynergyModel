# cognitive_synergy/models/prediction_heads.py
"""
Task-Specific Prediction Heads.

This module defines various prediction heads that can be attached to the
outputs of the CognitiveSynergyModel (e.g., world_state, projected embeddings)
to perform specific downstream tasks like VQA, Image Captioning, or
Image-Text Matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

# ==============================================================================
# Image-Text Matching Head (Previously defined - functional)
# ==============================================================================

class ImageTextMatchingHead(nn.Module):
    """
    A simple prediction head for Image-Text Matching (ITM).

    Takes combined features (e.g., concatenated projected CLS tokens, or world_state)
    and predicts whether the image and text correspond (binary classification).
    """
    def __init__(self, input_dim: int):
        """
        Args:
            input_dim (int): The dimension of the input features used for prediction.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Input dimension for ITM head must be positive.")

        # MLP classifier: Linear -> ReLU -> Dropout -> Linear (output 1 logit)
        hidden_dim = max(input_dim // 2, 64) # Ensure a reasonable minimum hidden dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1), # Regularization
            nn.Linear(hidden_dim, 1) # Output a single logit
        )
        print(f"Initialized ImageTextMatchingHead with InputDim={input_dim}, HiddenDim={hidden_dim}")

    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Predicts the matching score (logit).

        Args:
            combined_features (torch.Tensor): Input features. Shape: (batch_size, input_dim).

        Returns:
            torch.Tensor: Logits for binary classification (batch_size, 1).
        """
        # Input validation
        if combined_features.ndim != 2:
             raise ValueError(f"ITM Head input tensor must be 2D (batch, dim), got shape {combined_features.shape}")
        if combined_features.shape[1] != self.classifier[0].in_features:
             raise ValueError(f"Input tensor dim ({combined_features.shape[1]}) does not match "
                              f"ITM classifier input dim ({self.classifier[0].in_features})")

        logits = self.classifier(combined_features)
        return logits

# ==============================================================================
# Refined VQA Prediction Head
# ==============================================================================

class VQAPredictionHead(nn.Module):
    """
    Refined Visual Question Answering prediction head.

    Treats VQA as a classification task over a predefined answer vocabulary.
    Uses a simple MLP classifier on the world_state representation.
    """
    def __init__(self, world_state_dim: int, num_answers: int, hidden_dim_multiplier: int = 2):
        """
        Args:
            world_state_dim (int): Dimension of the input world_state embedding.
            num_answers (int): Number of possible answers in the VQA dataset vocabulary.
            hidden_dim_multiplier (int): Multiplier for the hidden layer size relative to input.
        """
        super().__init__()
        # Input validation
        if world_state_dim <= 0 or num_answers <= 0:
             raise ValueError("VQA Head dimensions must be positive.")
        print(f"Initializing VQAPredictionHead: WorldStateDim={world_state_dim}, NumAnswers={num_answers}")

        # Refined MLP classifier: Linear -> LayerNorm -> GELU -> Dropout -> Linear
        hidden_dim = world_state_dim * hidden_dim_multiplier
        self.classifier = nn.Sequential(
            nn.Linear(world_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Add LayerNorm for stability
            nn.GELU(), # GELU activation common in transformers
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_answers) # Output logits over the answer vocabulary
        )
        print(f"  VQA Classifier MLP: {world_state_dim} -> {hidden_dim} -> {num_answers}")

    def forward(self, world_state: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Takes world_state and predicts answer logits.

        Args:
            world_state (torch.Tensor): The world state embedding (batch_size, world_state_dim).
            **kwargs: Catches any other potential inputs (not used).

        Returns:
            torch.Tensor: Logits over the answer vocabulary (batch_size, num_answers).
        """
        # Input validation
        if world_state.ndim != 2:
             raise ValueError(f"VQA Head expects 2D world_state (batch, dim), got shape {world_state.shape}")
        if world_state.shape[1] != self.classifier[0].in_features:
             raise ValueError(f"World state dim ({world_state.shape[1]}) doesn't match VQA classifier input ({self.classifier[0].in_features})")

        logits = self.classifier(world_state)
        return logits

# ==============================================================================
# Refined Causal Language Modeling Head (Simplified)
# ==============================================================================

class CausalLMPredictionHead(nn.Module):
    """
    Refined (but still simplified) Causal Language Modeling head.

    This head demonstrates how the world_state could be used to influence
    language generation, typically by predicting logits over the LLM's vocabulary.
    A full implementation would usually involve conditioning the LLM's decoder.
    This version projects the world_state directly to vocabulary logits.
    """
    def __init__(self, world_state_dim: int, llm_embedding_dim: int, llm_vocab_size: int):
        """
        Args:
            world_state_dim (int): Dimension of the input world_state embedding.
            llm_embedding_dim (int): The embedding dimension of the target LLM.
                                     Used for an intermediate projection.
            llm_vocab_size (int): Size of the language model's vocabulary.
        """
        super().__init__()
        # Input validation
        if world_state_dim <= 0 or llm_embedding_dim <= 0 or llm_vocab_size <= 0:
             raise ValueError("CausalLM Head dimensions must be positive.")
        print(f"Initializing CausalLMPredictionHead: WorldStateDim={world_state_dim}, LLMEmbedDim={llm_embedding_dim}, VocabSize={llm_vocab_size}")

        # Project world_state potentially to match LLM embedding dim, then to vocab size.
        # This simulates mapping the fused state into the language space before prediction.
        # Could also directly project world_state_dim -> llm_vocab_size.
        self.projection = nn.Sequential(
            nn.Linear(world_state_dim, llm_embedding_dim),
            nn.LayerNorm(llm_embedding_dim),
            nn.GELU(),
            nn.Linear(llm_embedding_dim, llm_vocab_size) # Final projection to vocabulary logits
        )
        print(f"  CausalLM Projection: {world_state_dim} -> {llm_embedding_dim} -> {llm_vocab_size}")
        # Note: In a real generative model, this head would likely be replaced by
        # conditioning mechanisms within the LLM decoder itself, or this projection
        # might generate an input embedding prefix for the decoder.

    def forward(self, world_state: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Takes world_state and predicts logits over the language model vocabulary.

        Args:
            world_state (torch.Tensor): The world state embedding (batch_size, world_state_dim).
            **kwargs: Catches any other potential inputs.

        Returns:
            torch.Tensor: Logits over the vocabulary (batch_size, llm_vocab_size).
                          Interpretation depends on task (e.g., predicting first token,
                          or summary state).
        """
        # Input validation
        if world_state.ndim != 2:
             raise ValueError(f"CausalLM Head expects 2D world_state (batch, dim), got shape {world_state.shape}")
        if world_state.shape[1] != self.projection[0].in_features:
             raise ValueError(f"World state dim ({world_state.shape[1]}) doesn't match CausalLM head input ({self.projection[0].in_features})")

        # Project the world state to vocabulary logits
        logits = self.projection(world_state)
        return logits


