# cognitive_synergy/data/datasets.py
"""
PyTorch Dataset classes for loading and preprocessing data.

This module defines Dataset classes compatible with PyTorch DataLoaders.
Includes ContrastiveImageTextDataset and a more fleshed-out VQADataset example.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image # Pillow library for image loading
from typing import List, Dict, Any, Callable, Optional
import collections

# ==============================================================================
# Contrastive Dataset (Previously defined - functional)
# ==============================================================================

class ContrastiveImageTextDataset(Dataset):
    """
    A PyTorch Dataset for image-text contrastive learning.

    Loads data based on a manifest file (e.g., JSON) listing image paths
    and corresponding captions. Applies provided image and text transforms.
    """
    def __init__(self,
                 manifest_path: str,
                 image_transform: Callable,
                 text_transform: Callable,
                 image_root: Optional[str] = None):
        super().__init__()
        print(f"Loading Contrastive dataset manifest from: {manifest_path}")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

        self.manifest_path = manifest_path
        if not callable(image_transform): raise TypeError("image_transform must be callable.")
        if not callable(text_transform): raise TypeError("text_transform must be callable.")
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.image_root = image_root

        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            raise IOError(f"Error reading or parsing manifest file {manifest_path}: {e}")

        if not isinstance(self.data, list): raise TypeError("Manifest must be a list.")
        if not self.data: print(f"Warning: Manifest file {manifest_path} is empty.")
        else:
            first_item = self.data[0]
            if not isinstance(first_item, dict): raise TypeError("Manifest items must be dictionaries.")
            req_keys = {"image_path", "caption"}
            if not req_keys.issubset(first_item.keys()): raise ValueError(f"Manifest items missing keys: {req_keys - set(first_item.keys())}")
            if not isinstance(first_item['image_path'], str): raise TypeError("'image_path' must be a string.")
            if not isinstance(first_item['caption'], str): raise TypeError("'caption' must be a string.")

        print(f"Loaded {len(self.data)} contrastive samples.")
        if len(self.data) == 0: print("Warning: Contrastive dataset is empty.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not 0 <= idx < len(self.data): raise IndexError(f"Index {idx} out of bounds.")
        try:
            sample_info = self.data[idx]
            raw_image_path = sample_info['image_path']
            caption = sample_info['caption']
        except Exception as e: raise IndexError(f"Error accessing data at index {idx}: {e}")

        if self.image_root: full_image_path = os.path.join(self.image_root, raw_image_path)
        else: full_image_path = raw_image_path

        try:
            image = Image.open(full_image_path).convert('RGB')
        except FileNotFoundError: raise FileNotFoundError(f"Image file not found: {full_image_path} (index {idx})")
        except Exception as e: raise IOError(f"Error loading image {full_image_path} (index {idx}): {e}")

        try:
            transformed_image = self.image_transform(image)
            if not isinstance(transformed_image, torch.Tensor): raise TypeError("Image transform error.")
        except Exception as e: raise RuntimeError(f"Image transform failed for {full_image_path}: {e}")

        try:
            text_data = self.text_transform(caption)
            if not isinstance(text_data, dict) or 'input_ids' not in text_data or 'attention_mask' not in text_data:
                 raise TypeError("Text transform must return dict with 'input_ids' and 'attention_mask'.")
            if not isinstance(text_data['input_ids'], torch.Tensor) or not isinstance(text_data['attention_mask'], torch.Tensor):
                 raise TypeError("Text transform outputs must be torch.Tensors.")
        except Exception as e: raise RuntimeError(f"Text transform failed for caption (index {idx}): {e}")

        input_ids = text_data['input_ids'].squeeze(0) if text_data['input_ids'].ndim > 1 else text_data['input_ids']
        attention_mask = text_data['attention_mask'].squeeze(0) if text_data['attention_mask'].ndim > 1 else text_data['attention_mask']

        return {"image": transformed_image, "input_ids": input_ids, "attention_mask": attention_mask}


# ==============================================================================
# Fleshed-Out VQA Dataset Example
# ==============================================================================

class VQADataset(Dataset):
    """
    PyTorch Dataset for Visual Question Answering (VQA).

    Assumes a manifest file listing image paths, questions, and corresponding
    human answers (e.g., similar to VQA v2 format). Requires an answer vocabulary
    mapping common answers to indices. Calculates target scores based on human
    answer agreement.

    Args:
        manifest_path (str): Path to the JSON manifest file. Expected items:
                             {"image_path": str, "question": str, "answers": List[Dict]}.
                             The "answers" list contains dicts like {"answer": str}.
        image_transform (Callable): Transform for input images.
        text_transform (Callable): Transform for input questions (tokenization).
        answer_to_index (Dict[str, int]): Mapping from answer strings to vocabulary indices.
        num_answers (int): The total size of the answer vocabulary (number of classes).
        image_root (Optional[str]): Root directory for image paths. Defaults to None.
        vqa_score_threshold (float): Confidence threshold for VQA target score calculation
                                     (e.g., min(1.0, num_matches / 3)). Defaults to 3.0 (denominator).
    """
    def __init__(self,
                 manifest_path: str,
                 image_transform: Callable,
                 text_transform: Callable,
                 answer_to_index: Dict[str, int],
                 num_answers: int, # Should match len(answer_to_index)
                 image_root: Optional[str] = None,
                 vqa_score_denominator: float = 3.0): # Denominator for VQA score calc (e.g., 3 for VQA v2)

        super().__init__()
        print(f"Loading VQA dataset manifest from: {manifest_path}")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        if not callable(image_transform): raise TypeError("image_transform must be callable.")
        if not callable(text_transform): raise TypeError("text_transform must be callable.")
        if not isinstance(answer_to_index, dict): raise TypeError("answer_to_index must be a dictionary.")
        if num_answers <= 0: raise ValueError("num_answers must be positive.")
        if vqa_score_denominator <= 0: raise ValueError("vqa_score_denominator must be positive.")

        self.manifest_path = manifest_path
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.answer_to_index = answer_to_index
        self.num_answers = num_answers
        self.image_root = image_root
        self.vqa_score_denominator = vqa_score_denominator

        # --- Load Data Manifest ---
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            raise IOError(f"Error reading or parsing VQA manifest file {manifest_path}: {e}")

        # --- Validate Manifest Structure ---
        if not isinstance(self.data, list): raise TypeError("VQA Manifest must be a list.")
        if not self.data: print(f"Warning: VQA Manifest file {manifest_path} is empty.")
        else:
            first_item = self.data[0]
            if not isinstance(first_item, dict): raise TypeError("VQA Manifest items must be dictionaries.")
            req_keys = {"image_path", "question", "answers"}
            if not req_keys.issubset(first_item.keys()): raise ValueError(f"VQA Manifest items missing keys: {req_keys - set(first_item.keys())}")
            if not isinstance(first_item['image_path'], str): raise TypeError("'image_path' must be str.")
            if not isinstance(first_item['question'], str): raise TypeError("'question' must be str.")
            if not isinstance(first_item['answers'], list): raise TypeError("'answers' must be a list.")
            # Optionally check answer format: e.g., first_item['answers'][0]['answer']

        print(f"Loaded {len(self.data)} VQA samples.")
        if len(self.data) == 0: print("Warning: VQA dataset is empty.")

    def __len__(self) -> int:
        return len(self.data)

    def _calculate_vqa_target_scores(self, human_answers: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Calculates target scores for answers based on human agreement.
        Uses the VQA accuracy formula: min(1.0, num_matches / denominator).

        Args:
            human_answers (List[Dict[str, Any]]): List of answer dicts, e.g., [{"answer": "yes"}, ...].

        Returns:
            torch.Tensor: A tensor of shape (num_answers,) with scores for each possible answer.
        """
        target_scores = torch.zeros(self.num_answers)
        answer_counts = collections.Counter()
        for ans_info in human_answers:
            answer_str = ans_info.get("answer") # Get answer string
            if answer_str and isinstance(answer_str, str):
                 answer_counts[answer_str.lower().strip()] += 1 # Count occurrences (case-insensitive)

        # Calculate score for each answer based on counts
        for answer, count in answer_counts.items():
            if answer in self.answer_to_index:
                answer_index = self.answer_to_index[answer]
                # Calculate score using VQA formula (min(1.0, count / denominator))
                score = min(1.0, count / self.vqa_score_denominator)
                target_scores[answer_index] = score

        return target_scores


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves and preprocesses the VQA sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the processed sample:
                - 'image': The transformed image tensor.
                - 'input_ids': Token IDs for the question.
                - 'attention_mask': Attention mask for the question.
                - 'answer_target': Tensor of target scores for each possible answer.
        """
        if not 0 <= idx < len(self.data): raise IndexError(f"Index {idx} out of bounds.")

        # --- Get Sample Metadata ---
        try:
            sample_info = self.data[idx]
            raw_image_path = sample_info['image_path']
            question = sample_info['question']
            human_answers = sample_info['answers'] # List of answer dicts
        except Exception as e:
            raise IndexError(f"Error accessing VQA data at index {idx}: {e}")

        # --- Load and Transform Image ---
        if self.image_root: full_image_path = os.path.join(self.image_root, raw_image_path)
        else: full_image_path = raw_image_path
        try:
            image = Image.open(full_image_path).convert('RGB')
        except FileNotFoundError: raise FileNotFoundError(f"Image file not found: {full_image_path} (index {idx})")
        except Exception as e: raise IOError(f"Error loading image {full_image_path} (index {idx}): {e}")

        try:
            transformed_image = self.image_transform(image)
            if not isinstance(transformed_image, torch.Tensor): raise TypeError("Image transform error.")
        except Exception as e: raise RuntimeError(f"Image transform failed for {full_image_path}: {e}")

        # --- Transform Question Text ---
        try:
            question_data = self.text_transform(question)
            if not isinstance(question_data, dict) or 'input_ids' not in question_data or 'attention_mask' not in question_data:
                 raise TypeError("Text transform must return dict with 'input_ids' and 'attention_mask'.")
            if not isinstance(question_data['input_ids'], torch.Tensor) or not isinstance(question_data['attention_mask'], torch.Tensor):
                 raise TypeError("Text transform outputs must be torch.Tensors.")
        except Exception as e: raise RuntimeError(f"Text transform failed for question (index {idx}): {e}")

        # --- Calculate Answer Target Scores ---
        try:
            answer_target_scores = self._calculate_vqa_target_scores(human_answers)
        except Exception as e:
            print(f"Error calculating VQA target scores for index {idx}: {e}")
            # Return zeros or raise error depending on desired behavior
            answer_target_scores = torch.zeros(self.num_answers)
            # raise RuntimeError(f"Failed to calculate VQA target scores for index {idx}")


        # --- Prepare Output Dictionary ---
        input_ids = question_data['input_ids'].squeeze(0) if question_data['input_ids'].ndim > 1 else question_data['input_ids']
        attention_mask = question_data['attention_mask'].squeeze(0) if question_data['attention_mask'].ndim > 1 else question_data['attention_mask']

        output = {
            "image": transformed_image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_target": answer_target_scores, # Target scores for loss (e.g., BCEWithLogitsLoss)
            # Optionally include raw question/answers for debugging or specific eval metrics
            # "question_str": question,
            # "raw_answers": human_answers,
        }

        return output


