# cognitive_synergy/training/losses.py
"""
Loss function definitions for training the Cognitive Synergy Model.

Includes the contrastive alignment loss and potentially other task-specific
losses (e.g., for VQA, ITM, Causal LM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

# ==============================================================================
# Contrastive Alignment Loss (InfoNCE style)
# ==============================================================================

class ContrastiveAlignmentLoss(nn.Module):
    """
    Computes the contrastive alignment loss (InfoNCE style) between two sets of embeddings.

    Encourages embeddings from matched pairs (e.g., vision and language from the
    same sample) to be closer in the embedding space than embeddings from
    mismatched pairs within the batch. Assumes input embeddings are suitable
    for cosine similarity (e.g., projected to a common space and L2 normalized).
    """
    def __init__(self, temperature: float = 0.07, symmetric: bool = True):
        """
        Args:
            temperature (float): Temperature scaling factor for the similarity scores.
                                 Smaller values lead to sharper distributions. Must be positive.
            symmetric (bool): If True, computes the loss symmetrically
                              (vision -> language and language -> vision) and averages them.
                              If False, only computes vision -> language loss.
        """
        super().__init__()
        # Validate temperature
        if not isinstance(temperature, (float, int)) or temperature <= 0:
            raise ValueError("Temperature must be a positive number.")
        # Register temperature as a buffer (saved with model state, not trained by default)
        # Use register_buffer for non-parameter tensors that should be part of the state_dict
        self.register_buffer("temperature", torch.tensor(temperature))
        self.symmetric = symmetric
        print(f"Initialized ContrastiveAlignmentLoss: temperature={self.temperature.item():.3f}, symmetric={symmetric}")

    def forward(self, vision_embeddings: torch.Tensor, language_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates the contrastive alignment loss.

        Args:
            vision_embeddings (torch.Tensor): Batch of vision embeddings
                                             (batch_size, embedding_dim). It's recommended to L2 normalize
                                             these embeddings *before* passing them to this function.
            language_embeddings (torch.Tensor): Batch of language embeddings
                                                (batch_size, embedding_dim). It's recommended to L2 normalize
                                                these embeddings *before* passing them to this function.

        Returns:
            torch.Tensor: The calculated contrastive loss (scalar).
        """
        # --- Input Validation ---
        if vision_embeddings.ndim != 2 or language_embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D tensors (batch_size, embedding_dim). "
                             f"Got shapes: {vision_embeddings.shape}, {language_embeddings.shape}")
        if vision_embeddings.shape[0] != language_embeddings.shape[0]:
            raise ValueError("Batch sizes of vision and language embeddings must match. "
                             f"Got: {vision_embeddings.shape[0]} vs {language_embeddings.shape[0]}")
        if vision_embeddings.shape[1] != language_embeddings.shape[1]:
            raise ValueError("Embedding dimensions of vision and language embeddings must match. "
                             f"Got: {vision_embeddings.shape[1]} vs {language_embeddings.shape[1]}")

        batch_size = vision_embeddings.shape[0]
        embedding_dim = vision_embeddings.shape[1]
        device = vision_embeddings.device

        # Contrastive loss requires multiple samples (batch_size > 1) to compare against negatives.
        if batch_size <= 1:
            # Return 0 loss for batch size 1, as no negatives exist. Add warning.
            print(f"Warning: Contrastive loss calculated with batch size {batch_size}. "
                  "Requires batch size > 1 for negative samples. Returning 0 loss.")
            # Return a tensor on the correct device with requires_grad=True if inputs require grad
            # This prevents errors in backward pass when loss is not computed.
            req_grad = vision_embeddings.requires_grad or language_embeddings.requires_grad
            return torch.tensor(0.0, device=device, requires_grad=req_grad)

        # --- L2 Normalize Embeddings (Crucial for Cosine Similarity) ---
        # Ensure numerical stability by adding a small epsilon before sqrt during normalization.
        # It's often preferred to normalize in the projection head before this module.
        vision_embeddings_norm = F.normalize(vision_embeddings, p=2, dim=-1, eps=1e-12)
        language_embeddings_norm = F.normalize(language_embeddings, p=2, dim=-1, eps=1e-12)

        # --- Calculate Cosine Similarity Matrix ---
        # vision_embeddings_norm: [batch_size, embedding_dim]
        # language_embeddings_norm: [batch_size, embedding_dim]
        # similarity_matrix: [batch_size, batch_size]
        # Element (i, j) is the cosine similarity between the i-th vision embedding
        # and the j-th language embedding.
        similarity_matrix = torch.matmul(vision_embeddings_norm, language_embeddings_norm.t())

        # Scale similarities by temperature
        # Clamping temperature just in case, although validated in init
        # Access buffer correctly using self.temperature
        temp = torch.clamp(self.temperature, min=1e-6) # Ensure temperature is positive
        logits = similarity_matrix / temp # Logits for the classification task

        # --- Create Ground Truth Labels ---
        # The diagonal elements correspond to matched pairs (positive samples).
        # Create labels representing the indices [0, 1, 2, ..., batch_size-1]
        labels = torch.arange(batch_size, dtype=torch.long, device=device)

        # --- Calculate Loss using Cross-Entropy ---
        # Calculate cross-entropy loss for vision -> language alignment
        # Treats rows of logits as scores for predicting the correct language embedding (column index)
        # for each vision embedding (row index). The target is the index of the matching language embedding.
        loss_v_to_l = F.cross_entropy(logits, labels)

        if self.symmetric:
            # Calculate cross-entropy loss for language -> vision alignment
            # Transpose logits: rows become language embeddings, columns become vision embeddings.
            # Treats rows of logits.t() as scores for predicting the correct vision embedding (column index)
            # for each language embedding (row index). The target is the index of the matching vision embedding.
            loss_l_to_v = F.cross_entropy(logits.t(), labels)
            # Average the two directional losses
            loss = (loss_v_to_l + loss_l_to_v) / 2.0
        else:
            # Use only the vision -> language loss
            loss = loss_v_to_l

        return loss

# ==============================================================================
# Placeholder/Example Task-Specific Losses
# ==============================================================================

def image_text_matching_loss(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Calculates Binary Cross-Entropy loss for Image-Text Matching.

    Args:
        logits (torch.Tensor): Raw output logits from the ITM prediction head (batch_size, 1) or (batch_size,).
        targets (torch.Tensor): Ground truth labels (batch_size), where 1 indicates a match
                                and 0 indicates a mismatch. Should be float type for BCEWithLogitsLoss.

    Returns:
        torch.Tensor: The calculated BCE loss (scalar).
    """
    # Ensure logits and targets have compatible shapes (batch_size,)
    if logits.ndim > 1:
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1) # Ensure logits are [batch_size] if shape is [batch_size, 1]
        else:
            raise ValueError(f"ITM logits shape unexpected: {logits.shape}. Expected (batch,) or (batch, 1).")
    if targets.ndim > 1:
         raise ValueError(f"ITM targets shape unexpected: {targets.shape}. Expected (batch,).")
    if logits.shape != targets.shape:
        raise ValueError(f"Logits shape {logits.shape} must match targets shape {targets.shape} for ITM BCE loss.")

    # BCEWithLogitsLoss combines Sigmoid layer and BCELoss in one single class.
    # This version is more numerically stable than using a plain Sigmoid followed by BCELoss.
    # Ensure targets are float type as required by the loss function.
    loss = F.binary_cross_entropy_with_logits(logits, targets.float())
    return loss


def vqa_loss(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Calculates loss for Visual Question Answering based on target scores.

    Uses Binary Cross-Entropy with Logits, suitable when targets represent
    soft scores (e.g., based on human agreement) rather than single class indices.

    Args:
        logits (torch.Tensor): Raw output logits from the VQA prediction head
                               (batch_size, num_answers).
        targets (torch.Tensor): Ground truth answer scores (batch_size, num_answers).
                                Values typically between 0 and 1.

    Returns:
        torch.Tensor: The calculated BCE loss (scalar).
    """
    # Assumes targets are soft scores (e.g., from VQADataset calculation)
    if logits.ndim != 2:
         raise ValueError(f"VQA logits must be 2D (batch, num_answers), got shape {logits.shape}")
    if targets.ndim != 2:
         raise ValueError(f"VQA targets (scores) must be 2D (batch, num_answers), got shape {targets.shape}")
    if logits.shape != targets.shape:
         raise ValueError(f"Shape mismatch between VQA logits ({logits.shape}) and targets ({targets.shape})")

    # Use BCEWithLogitsLoss for multi-label classification with soft targets
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


# --- Loss Function Factory (Optional but useful) ---
# Can be useful for selecting the loss based on config in the trainer

def get_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to get the primary loss function based on configuration.

    Args:
        config (Dict): Configuration dictionary, expected to have a 'loss' section
                       with 'type' and potentially other parameters like 'temperature'.

    Returns:
        nn.Module: The instantiated loss function module.
    """
    loss_config = config.get('loss', {})
    if not isinstance(loss_config, dict):
        raise TypeError(f"Expected 'loss' section in config to be a dictionary, got {type(loss_config)}")

    loss_type = loss_config.get('type', None)
    if loss_type is None:
         # Default to contrastive if type is not specified
         print("Warning: Loss type ('loss.type') not specified. Defaulting to 'contrastive_alignment'.")
         loss_type = 'contrastive_alignment'

    loss_type = loss_type.lower()
    print(f"Creating loss function of type: {loss_type}")

    if loss_type == 'contrastive_alignment':
        temperature = loss_config.get('temperature', 0.07)
        symmetric = loss_config.get('symmetric', True)
        return ContrastiveAlignmentLoss(temperature=temperature, symmetric=symmetric)
    # --- Add other primary loss types here if the main objective changes ---
    # Example: If the main task becomes VQA using BCE loss
    # elif loss_type == 'vqa_bce':
    #     # This loss doesn't have parameters like temperature, so just return the function
    #     # Need to wrap it in a simple nn.Module if the trainer expects a module instance
    #     class VQALossWrapper(nn.Module):
    #          def forward(self, logits, targets):
    #              return vqa_loss(logits, targets)
    #     return VQALossWrapper()

    else:
        # Raise error for unsupported primary loss types
        raise ValueError(f"Unsupported primary loss type in config: '{loss_type}'. Supported: 'contrastive_alignment'.")

    # Note: Auxiliary losses (like ITM, VQA when contrastive is primary) are typically handled
    # separately in the Trainer's training step, potentially using the functional forms defined above.


