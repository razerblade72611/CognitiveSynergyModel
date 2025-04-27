# cognitive_synergy/data/transforms.py
"""
Data Transformation Pipelines for Images and Text.

This module defines functions or classes to create preprocessing pipelines
for image and text data, compatible with the Dataset classes.
"""

import torch
import torchvision.transforms as T
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Dict, Tuple, Optional, Union, List

# ==============================================================================
# Image Transformations
# ==============================================================================

def get_image_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), # ImageNet mean
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),  # ImageNet std
    is_train: bool = False
) -> T.Compose:
    """
    Creates a standard image transformation pipeline using torchvision.

    Args:
        image_size (int): Target size for the image (e.g., 224 for ViT base).
        mean (Tuple[float, float, float]): Mean values for normalization.
        std (Tuple[float, float, float]): Standard deviation values for normalization.
        is_train (bool): If True, applies training augmentations (RandomResizedCrop, RandomHorizontalFlip).
                         If False, applies validation/test transformations (Resize, CenterCrop).

    Returns:
        T.Compose: A torchvision composition of transformations.
    """
    print(f"Creating image transform: size={image_size}, is_train={is_train}")
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if not (isinstance(mean, (list, tuple)) and len(mean) == 3):
         raise TypeError("mean must be a tuple or list of 3 floats.")
    if not (isinstance(std, (list, tuple)) and len(std) == 3):
         raise TypeError("std must be a tuple or list of 3 floats.")

    if is_train:
        # Training transformations: includes augmentation for robustness
        transform = T.Compose([
            # Randomly resize and crop to the target size. Scale determines the range of area cropped.
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            # Randomly flip the image horizontally with a 50% probability.
            T.RandomHorizontalFlip(p=0.5),
            # Convert PIL image (H, W, C) or numpy array to tensor (C, H, W) in range [0, 1].
            T.ToTensor(),
            # Normalize tensor image with mean and standard deviation.
            T.Normalize(mean=mean, std=std)
        ])
        print("  Applied training augmentations (RandomResizedCrop, RandomHorizontalFlip).")
    else:
        # Validation/Testing transformations: deterministic, no random augmentation
        transform = T.Compose([
            # Resize the smaller edge to image_size while maintaining aspect ratio.
            # Using 256 for resize before 224 crop is common practice, but direct resize works too.
            # Let's use direct resize for simplicity here matching common ViT preprocessing.
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            # Crop the center square of the specified size.
            T.CenterCrop(image_size),
            # Convert to tensor.
            T.ToTensor(),
            # Normalize.
            T.Normalize(mean=mean, std=std)
        ])
        print("  Applied validation/test transformations (Resize, CenterCrop).")

    return transform

# ==============================================================================
# Text Transformations
# ==============================================================================

class TextTransform:
    """
    A callable class for text transformation using a Hugging Face tokenizer.

    Tokenizes text, adds special tokens, pads/truncates, and returns tensors
    suitable for input to models like BERT.
    """
    def __init__(self, tokenizer_name: str = 'bert-base-uncased', max_length: int = 128):
        """
        Args:
            tokenizer_name (str): Name or path of the Hugging Face tokenizer to load.
            max_length (int): Maximum sequence length for padding/truncation.
        """
        print(f"Initializing TextTransform with tokenizer: {tokenizer_name}, max_length: {max_length}")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        try:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
             print(f"Error loading tokenizer '{tokenizer_name}': {e}")
             raise e # Re-raise after logging
        self.max_length = max_length
        print("Tokenizer loaded successfully.")

    def __call__(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        Applies tokenization and preprocessing to input text.

        Args:
            text (Union[str, List[str]]): A single string or a list of strings to tokenize.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_ids': Tensor of token IDs.
                - 'attention_mask': Tensor of attention masks.
                Shape is typically (batch_size, max_length) if input is list,
                or (1, max_length) if input is single string (before potential squeeze in Dataset).
        """
        # Handle potential None or empty string input gracefully
        if text is None:
             print("Warning: Received None text input in TextTransform. Processing as empty string.")
             text_to_process = "" # Process empty string to get padding tokens
        elif isinstance(text, str) and not text.strip():
             print("Warning: Received empty string input in TextTransform.")
             text_to_process = text # Process empty string
        elif isinstance(text, list) and not any(t and t.strip() for t in text):
             print("Warning: Received list of empty strings in TextTransform.")
             text_to_process = text # Process list of empty strings
        else:
             text_to_process = text

        # Tokenize the text using the loaded tokenizer
        # - add_special_tokens=True: Adds [CLS], [SEP] (for BERT-like models)
        # - max_length: Specifies the target sequence length
        # - padding='max_length': Pads shorter sequences to max_length
        # - truncation=True: Truncates longer sequences to max_length
        # - return_tensors='pt': Returns PyTorch tensors
        try:
            tokenized_output = self.tokenizer(
                text_to_process,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt', # Return PyTorch tensors
                return_attention_mask=True # Ensure attention mask is returned
            )
        except Exception as e:
             # Log safely, avoid printing potentially large text input directly
             text_repr = str(text_to_process)[:100] + ('...' if len(str(text_to_process)) > 100 else '')
             print(f"Error during tokenization for text: '{text_repr}'. Error: {e}")
             raise e # Re-raise after logging

        # The tokenizer output is already a dictionary containing 'input_ids' and 'attention_mask'
        # Tensors will have shape [1, max_length] for single string input, or [batch_size, max_length] for list input.
        # The Dataset __getitem__ is responsible for squeezing the batch dimension if needed.
        return {
            "input_ids": tokenized_output['input_ids'],
            "attention_mask": tokenized_output['attention_mask']
        }

# Example Usage (for testing purposes when running this file directly)
if __name__ == "__main__":
    print("\n--- Testing Image Transform (Validation) ---")
    val_transform = get_image_transform(is_train=False)
    print(val_transform)
    # Requires PIL installed: pip install Pillow
    try:
        from PIL import Image
        dummy_image = Image.new('RGB', (300, 400), color = 'red')
        transformed_val = val_transform(dummy_image)
        print("Validation image tensor shape:", transformed_val.shape) # Should be [3, 224, 224]
    except ImportError:
        print("Pillow not installed, skipping image transform test.")
    except Exception as e:
        print(f"Error during image transform test: {e}")


    print("\n--- Testing Image Transform (Training) ---")
    train_transform = get_image_transform(is_train=True)
    print(train_transform)
    try:
        from PIL import Image
        dummy_image = Image.new('RGB', (300, 400), color = 'blue')
        transformed_train = train_transform(dummy_image)
        print("Training image tensor shape:", transformed_train.shape) # Should be [3, 224, 224]
    except ImportError:
        print("Pillow not installed, skipping image transform test.")
    except Exception as e:
        print(f"Error during image transform test: {e}")


    print("\n--- Testing Text Transform ---")
    try:
        text_transform = TextTransform(tokenizer_name='bert-base-uncased', max_length=32)
        sample_text = "This is an example sentence for BERT."
        tokenized = text_transform(sample_text)
        print("Sample Text:", sample_text)
        print("Tokenized Output Keys:", tokenized.keys())
        print("Input IDs:", tokenized['input_ids'])
        print("Attention Mask:", tokenized['attention_mask'])
        print("Input IDs Shape:", tokenized['input_ids'].shape) # Should be [1, 32]

        # Test squeezing if needed by dataset
        print("Input IDs Squeezed Shape:", tokenized['input_ids'].squeeze(0).shape) # Should be [32]

        print("\n--- Testing Text Transform with Batch ---")
        sample_batch = ["First sentence.", "Second, slightly longer sentence."]
        tokenized_batch = text_transform(sample_batch)
        print("Sample Batch:", sample_batch)
        print("Batch Input IDs Shape:", tokenized_batch['input_ids'].shape) # Should be [2, 32]
        print("Batch Attention Mask Shape:", tokenized_batch['attention_mask'].shape) # Should be [2, 32]

        print("\n--- Testing Text Transform with Empty Input ---")
        tokenized_empty = text_transform("")
        print("Empty string Input IDs:", tokenized_empty['input_ids'])
        print("Empty string Attention Mask:", tokenized_empty['attention_mask'])

    except Exception as e:
        print(f"Error during text transform test (is 'transformers' installed?): {e}")


