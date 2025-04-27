# cognitive_synergy/utils/metrics.py
"""
Evaluation Metrics Calculation.

Defines functions to compute various evaluation metrics relevant to
multimodal tasks like VQA, Image-Text Matching, Retrieval, etc.
"""

import torch
from typing import Dict, List, Any, Optional
import collections
import logging # Use standard logging

# Get a logger instance
logger = logging.getLogger("CognitiveSynergy.Metrics") # Use hierarchical naming


# ==============================================================================
# Basic Accuracy Metric
# ==============================================================================

def calculate_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 1
) -> float:
    """
    Calculates top-k accuracy for classification tasks.

    Args:
        logits (torch.Tensor): Model output logits (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels (batch_size), expected as class indices (long tensor).
        top_k (int): The 'k' in top-k accuracy. Defaults to 1 (top-1 accuracy).

    Returns:
        float: The calculated top-k accuracy value (between 0.0 and 1.0).
    """
    # Input validation
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
         raise TypeError("Logits and targets must be torch tensors.")
    if logits.ndim != 2:
        raise ValueError(f"Logits must be 2D (batch, num_classes), got shape {logits.shape}")
    if targets.ndim != 1:
        raise ValueError(f"Targets must be 1D (batch), got shape {targets.shape}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(f"Batch size mismatch between logits ({logits.shape[0]}) and targets ({targets.shape[0]})")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    num_classes = logits.shape[1]
    if top_k > num_classes:
         logger.warning(f"top_k ({top_k}) is greater than the number of classes ({num_classes}). Setting top_k to {num_classes}.")
         top_k = num_classes

    batch_size = targets.size(0)
    if batch_size == 0:
        logger.warning("Calculating accuracy with empty batch (size 0). Returning 0.0.")
        return 0.0

    with torch.no_grad(): # Ensure no gradients are computed for metric calculation
        # Get the top k predictions for each sample
        # `torch.topk` returns (values, indices)
        # Ensure inputs are on the same device implicitly
        _values, pred_indices = torch.topk(logits, k=top_k, dim=1, largest=True, sorted=True) # Shape: [batch_size, top_k]

        # Reshape targets to [batch_size, 1] for comparison broadcasting
        targets_reshaped = targets.view(-1, 1)

        # Check if the true target index is present in the top k predicted indices
        # `eq` performs element-wise equality check. `sum(dim=1)` checks if *any* of the top k match.
        # Ensure comparison happens with tensors on the same device.
        correct = pred_indices.eq(targets_reshaped).sum(dim=1) # Shape: [batch_size], 1 if correct, 0 otherwise

        # Calculate accuracy
        accuracy = correct.float().sum().item() / batch_size

    return accuracy

# ==============================================================================
# Retrieval Recall@K Metric
# ==============================================================================

def calculate_retrieval_recall_at_k(
    similarity_matrix: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calculates Recall@K for image-text retrieval tasks.

    Assumes a similarity matrix where rows are queries (e.g., images) and
    columns are candidates (e.g., texts), and the diagonal represents
    the correct matches within the batch.

    Args:
        similarity_matrix (torch.Tensor): Cosine similarity matrix (N, N).
        k_values (List[int]): List of K values for which to calculate recall.

    Returns:
        Dict[str, float]: Dictionary mapping 'Recall@K' to its calculated value (0.0 to 1.0).
    """
    # Input validation
    if not isinstance(similarity_matrix, torch.Tensor):
         raise TypeError("Similarity matrix must be a torch tensor.")
    if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError(f"Similarity matrix must be square for this simple Recall@K calculation. Got shape {similarity_matrix.shape}")
    if not k_values or not isinstance(k_values, list) or not all(isinstance(k, int) and k > 0 for k in k_values):
         raise ValueError("k_values must be a list of positive integers.")

    num_queries = similarity_matrix.shape[0]
    if num_queries == 0:
        logger.warning("Calculating retrieval recall with empty similarity matrix (size 0). Returning 0.0 for all K.")
        # Return 0 for all k if the batch is empty
        return {f"Recall@{k}": 0.0 for k in k_values}

    results = {}
    with torch.no_grad():
        # Get the indices of the candidates sorted by similarity for each query
        # Sort descending (highest similarity first)
        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True) # Shape: [num_queries, num_candidates]

        # Ground truth: the correct candidate index for query i is i
        ground_truth = torch.arange(num_queries, device=similarity_matrix.device)

        for k in sorted(list(set(k_values))): # Ensure unique sorted k values
            # Cap k at the number of candidates available
            k_capped = min(k, num_queries)
            if k > num_queries:
                 logger.warning(f"k={k} is larger than the number of candidates ({num_queries}). Calculating Recall@{k_capped} instead.")

            # Check if the ground truth index appears within the top K retrieved indices
            # Take top k indices: sorted_indices[:, :k_capped]
            # Compare with ground truth reshaped for broadcasting: ground_truth.view(-1, 1)
            correct_at_k = (sorted_indices[:, :k_capped] == ground_truth.view(-1, 1)).sum(dim=1) # [num_queries], 1 if correct within top k, 0 otherwise
            recall_at_k = correct_at_k.float().sum().item() / num_queries
            results[f"Recall@{k}"] = recall_at_k # Store result with original k key

    return results


# ==============================================================================
# VQA Accuracy (VQA v2 Style)
# ==============================================================================

def calculate_vqa_accuracy(
    batch_predictions: List[str],
    batch_ground_truth_answers: List[List[Dict[str, Any]]],
    vqa_score_denominator: float = 3.0 # Denominator for VQA score calc (e.g., 3 for VQA v2)
) -> float:
    """
    Calculates accuracy for VQA tasks based on the VQA v2 evaluation metric.

    Compares the predicted answer string against a list of human ground truth
    answers and assigns a score based on agreement level.

    Args:
        batch_predictions (List[str]): A list of predicted answer strings for the batch.
                                       Length should be batch_size.
        batch_ground_truth_answers (List[List[Dict[str, Any]]]):
                                       A list (batch_size) where each element is a list
                                       of ground truth answer dictionaries for one question.
                                       Example inner list: [{"answer": "yes"}, {"answer": "Yes"}, ...].
        vqa_score_denominator (float): The denominator used in the VQA accuracy formula
                                      (e.g., 3 corresponds to min(1.0, num_matches / 3)).

    Returns:
        float: The calculated average VQA accuracy for the batch (between 0.0 and 1.0).
    """
    # Input validation
    if not isinstance(batch_predictions, list) or not isinstance(batch_ground_truth_answers, list):
         raise TypeError("Predictions and ground truth answers must be lists.")
    if len(batch_predictions) != len(batch_ground_truth_answers):
        raise ValueError(f"Batch size mismatch between predictions ({len(batch_predictions)}) "
                         f"and ground truth answers ({len(batch_ground_truth_answers)}).")
    if vqa_score_denominator <= 0:
        raise ValueError("vqa_score_denominator must be positive.")

    batch_size = len(batch_predictions)
    if batch_size == 0:
        logger.warning("Calculating VQA accuracy with empty batch (size 0). Returning 0.0.")
        return 0.0

    total_accuracy_score = 0.0

    for i in range(batch_size):
        pred_answer = batch_predictions[i]
        gt_answers_list = batch_ground_truth_answers[i]

        # Validate item types
        if not isinstance(pred_answer, str):
             logger.warning(f"Prediction at index {i} is not a string ({type(pred_answer)}). Assigning score 0.")
             accuracy_score = 0.0
        elif not isinstance(gt_answers_list, list):
             logger.warning(f"Ground truth answers at index {i} is not a list ({type(gt_answers_list)}). Assigning score 0.")
             accuracy_score = 0.0
        else:
            # Process valid prediction and ground truth
            pred_answer_processed = pred_answer.lower().strip()

            # Count occurrences of each ground truth answer (case-insensitive)
            answer_counts = collections.Counter()
            for ans_info in gt_answers_list:
                # Safely get the answer string from the dictionary
                answer_str = ans_info.get("answer")
                if answer_str and isinstance(answer_str, str):
                     answer_counts[answer_str.lower().strip()] += 1

            # Check if the predicted answer matches any of the ground truth answers
            num_matches = answer_counts.get(pred_answer_processed, 0)

            # Calculate VQA accuracy score for this sample: min(1.0, # humans that provided that answer / denominator)
            accuracy_score = min(1.0, num_matches / vqa_score_denominator)

        total_accuracy_score += accuracy_score

    # Average accuracy over the batch
    average_batch_accuracy = total_accuracy_score / batch_size if batch_size > 0 else 0.0

    return average_batch_accuracy


# --- Helper function to get predicted answer string from logits ---
# This would typically be used in the evaluation loop before calling calculate_vqa_accuracy

def get_predicted_answers_from_logits(
    logits: torch.Tensor,
    index_to_answer: Dict[int, str]
) -> List[str]:
    """
    Converts output logits to predicted answer strings using an index-to-answer mapping.

    Args:
        logits (torch.Tensor): Model output logits (batch_size, num_answers).
        index_to_answer (Dict[int, str]): Mapping from vocabulary index to answer string.

    Returns:
        List[str]: A list of predicted answer strings for the batch.
    """
    # Input validation
    if not isinstance(logits, torch.Tensor): raise TypeError("Logits must be a torch tensor.")
    if logits.ndim != 2: raise ValueError(f"Logits must be 2D (batch, num_answers), got shape {logits.shape}")
    if not index_to_answer or not isinstance(index_to_answer, dict):
         raise ValueError("index_to_answer mapping must be a non-empty dictionary.")
    if logits.shape[0] == 0:
         return [] # Return empty list for empty batch

    with torch.no_grad():
        # Get index of highest logit -> predicted answer index
        pred_indices = torch.argmax(logits, dim=1)
        # Map index to string, using "[UNK]" if index is not in the map
        predicted_answers = [index_to_answer.get(idx.item(), "[UNK]") for idx in pred_indices]

    return predicted_answers


# Example Usage (when running this file directly)
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)-8s | %(name)s | %(message)s')

    print("--- Testing Metrics ---")

    # Accuracy Example
    print("\n1. Accuracy Calculation")
    logits_acc = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    targets_acc = torch.tensor([1, 0, 1, 2]) # Ground truth indices
    acc1 = calculate_accuracy(logits_acc, targets_acc, top_k=1)
    acc2 = calculate_accuracy(logits_acc, targets_acc, top_k=2)
    acc_high_k = calculate_accuracy(logits_acc, targets_acc, top_k=5) # Test k > num_classes
    print(f"  Targets: {targets_acc}")
    print(f"  Logits:\n{logits_acc}")
    print(f"  Top-1 Accuracy: {acc1:.4f} (Expected: 0.75)")
    print(f"  Top-2 Accuracy: {acc2:.4f} (Expected: 1.00)")
    print(f"  Top-5 Accuracy (capped at 3): {acc_high_k:.4f} (Expected: 1.00)")

    # Retrieval Example
    print("\n2. Retrieval Recall@K Calculation")
    sim_matrix = torch.tensor([
        [0.9, 0.2, 0.1, 0.3], # Query 0 -> Correct is 0 (score 0.9) -> R@1 Match
        [0.1, 0.8, 0.3, 0.2], # Query 1 -> Correct is 1 (score 0.8) -> R@1 Match
        [0.4, 0.7, 0.3, 0.1], # Query 2 -> Correct is 2 (score 0.3) -> R@1 Miss, R@2 Hit
        [0.6, 0.1, 0.3, 0.2]  # Query 3 -> Correct is 3 (score 0.2) -> R@1 Miss, R@2 Miss, R@3 Hit
    ])
    recall_results = calculate_retrieval_recall_at_k(sim_matrix, k_values=[1, 2, 3, 5])
    print(f"  Similarity Matrix:\n{sim_matrix}")
    print(f"  Recall Results: {recall_results}") # Expected R@1=0.5, R@2=0.75, R@3=1.0, R@5=1.0 (capped at R@4)

    # VQA Accuracy Example
    print("\n3. VQA Accuracy Calculation")
    # Example data (batch size 3)
    batch_preds = ["cat", "yes", "2"]
    batch_gts = [
        [{"answer": "cat"}, {"answer": "CAT"}, {"answer": "feline"}], # Sample 1: pred "cat" matches 2/3 -> score 0.666...
        [{"answer": "no"}, {"answer": "no"}, {"answer": "No"}],      # Sample 2: pred "yes" matches 0/3 -> score 0.0
        [{"answer": "2"}, {"answer": "2"}, {"answer": "2"}, {"answer": "two"}, {"answer": "2"}] # Sample 3: pred "2" matches 4/3 -> score 1.0
    ]
    vqa_acc = calculate_vqa_accuracy(batch_preds, batch_gts, vqa_score_denominator=3.0)
    expected_vqa_acc = (min(1.0, 2/3) + min(1.0, 0/3) + min(1.0, 4/3)) / 3
    print(f"  Predicted Answers: {batch_preds}")
    # print(f"  Ground Truth Answers: {batch_gts}") # Can be long
    print(f"  Calculated VQA Accuracy: {vqa_acc:.4f}")
    print(f"  Expected VQA Accuracy:   {expected_vqa_acc:.4f}")

    # Example using helper function
    print("\n4. Getting Predicted Answers from Logits")
    logits_vqa_test = torch.tensor([
        [0.1, 0.8, 0.1], # Predict index 1 ('cat')
        [0.9, 0.0, 0.1], # Predict index 0 ('yes')
        [0.1, 0.2, 0.7]  # Predict index 2 ('2')
    ])
    idx_to_ans_map = {0: "yes", 1: "cat", 2: "2"}
    predicted_strings = get_predicted_answers_from_logits(logits_vqa_test, idx_to_ans_map)
    print(f"  Logits:\n{logits_vqa_test}")
    print(f"  Index to Answer Map: {idx_to_ans_map}")
    print(f"  Predicted Answer Strings: {predicted_strings}") # Expected: ['cat', 'yes', '2']


