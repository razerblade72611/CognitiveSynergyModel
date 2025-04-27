# cognitive_synergy/data/dataloaders.py
"""
Utility functions for creating PyTorch DataLoaders.
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from typing import Optional, Callable

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[Callable] = None,
    use_distributed: bool = False,
    world_size: int = 1,
    rank: int = 0
) -> DataLoader:
    """
    Creates a PyTorch DataLoader with appropriate sampling for standard or distributed training.

    Args:
        dataset (Dataset): The PyTorch Dataset instance to load data from.
        batch_size (int): How many samples per batch to load. Must be positive.
        num_workers (int): How many subprocesses to use for data loading.
                           0 means that the data will be loaded in the main process. Defaults to 0.
        shuffle (bool): Set to True to have the data reshuffled at every epoch (ignored if using DistributedSampler).
                        Defaults to False.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory
                           before returning them. Useful for faster GPU transfers. Defaults to True.
        drop_last (bool): Set to True to drop the last incomplete batch, if the dataset size
                          is not divisible by the batch size. Defaults to False.
        collate_fn (Optional[Callable]): Merges a list of samples to form a mini-batch of Tensor(s).
                                         Used when using batched loading from a map-style dataset.
                                         Defaults to None (uses default PyTorch collate_fn).
        use_distributed (bool): Set to True if using distributed training (e.g., DDP). Defaults to False.
        world_size (int): Number of processes in the distributed group (required if use_distributed=True).
                          Defaults to 1.
        rank (int): Rank of the current process in the distributed group (required if use_distributed=True).
                    Defaults to 0.

    Returns:
        DataLoader: The configured PyTorch DataLoader instance.
    """
    print(f"Creating DataLoader: batch_size={batch_size}, num_workers={num_workers}, shuffle={shuffle}, distributed={use_distributed}")

    # Input validation
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, but got {batch_size}")
    if num_workers < 0:
         raise ValueError(f"num_workers cannot be negative, got {num_workers}")

    sampler = None
    # Determine sampler based on distributed training flag
    if use_distributed:
        if not torch.distributed.is_available():
             raise RuntimeError("Distributed training requested but torch.distributed is not available.")
        if world_size <= 0:
             raise ValueError(f"world_size must be positive for distributed training, got {world_size}")
        if not (0 <= rank < world_size):
             raise ValueError(f"rank ({rank}) must be between 0 and world_size-1 ({world_size-1}) for distributed training.")

        # DistributedSampler handles shuffling if shuffle=True and epoch is set externally
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)
        print(f"  Using DistributedSampler: world_size={world_size}, rank={rank}, shuffle={shuffle}, drop_last={drop_last}")
        # When using DistributedSampler, the shuffle argument in DataLoader itself must be False.
        effective_shuffle = False
    else:
        # Standard sampling: RandomSampler for training (if shuffle=True), SequentialSampler otherwise.
        if shuffle:
            sampler = RandomSampler(dataset)
            print("  Using RandomSampler.")
        else:
            sampler = SequentialSampler(dataset)
            print("  Using SequentialSampler.")
        effective_shuffle = False # Sampler handles shuffling, so DataLoader shuffle is False

    # Create the DataLoader instance
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # drop_last is primarily handled by the sampler in distributed mode if set there.
        # For standard mode, setting it here ensures the last batch is dropped if needed.
        drop_last=drop_last and not use_distributed,
        collate_fn=collate_fn, # Use default collate if None
        # shuffle must be False when a sampler is provided.
        shuffle=effective_shuffle
    )

    print("DataLoader created successfully.")
    return dataloader

# Example Usage (Conceptual - requires dataset object)
if __name__ == "__main__":
    print("\n--- Example DataLoader Creation ---")
    # Create a dummy dataset for demonstration
    class DummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            if not 0 <= idx < self.size:
                 raise IndexError
            # Example: return dict matching expected model input
            return {
                'image': torch.randn(3, 224, 224),
                'input_ids': torch.randint(0, 1000, (128,)),
                'attention_mask': torch.ones(128, dtype=torch.long)
            }

    dummy_ds = DummyDataset(size=105) # Use a size not divisible by common batch sizes
    print(f"Dummy dataset size: {len(dummy_ds)}")

    # Example 1: Simple validation loader
    print("\n1. Validation Loader (Sequential)")
    val_loader = create_dataloader(
        dataset=dummy_ds,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    print(f"   DataLoader length: {len(val_loader)}") # Expected: ceil(105/16) = 7
    try:
        batch = next(iter(val_loader))
        print(f"   First batch shapes: image={batch['image'].shape}, input_ids={batch['input_ids'].shape}")
    except Exception as e:
        print(f"   Error getting first batch: {e}")


    # Example 2: Training loader with shuffling, dropping last batch
    print("\n2. Training Loader (Shuffled, Drop Last)")
    train_loader = create_dataloader(
        dataset=dummy_ds,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    print(f"   DataLoader length: {len(train_loader)}") # Expected: floor(105/32) = 3
    try:
        batch = next(iter(train_loader))
        print(f"   First batch shapes: image={batch['image'].shape}, input_ids={batch['input_ids'].shape}")
    except Exception as e:
        print(f"   Error getting first batch: {e}")


    # Example 3: Distributed loader (conceptual)
    # Requires setting up torch.distributed environment first
    # print("\n3. Conceptual Distributed DataLoader creation (requires distributed setup):")
    # try:
    #     # Mock distributed setup for testing sampler creation
    #     os.environ['MASTER_ADDR'] = 'localhost'
    #     os.environ['MASTER_PORT'] = '29500'
    #     torch.distributed.init_process_group(backend='gloo', rank=0, world_size=2) # Example setup
    #
    #     dist_loader = create_dataloader(
    #         dataset=dummy_ds,
    #         batch_size=16,
    #         shuffle=True,
    #         num_workers=1,
    #         use_distributed=True,
    #         world_size=2, # Example world size
    #         rank=0 # Example rank
    #     )
    #     print(f"   Distributed DataLoader (rank 0) length: {len(dist_loader)}") # Expected: ceil(105/2) / 16 = ceil(53/16) = 4
    #     batch = next(iter(dist_loader))
    #     print(f"   First batch shapes (rank 0): image={batch['image'].shape}, input_ids={batch['input_ids'].shape}")
    #
    #     if torch.distributed.is_initialized():
    #          torch.distributed.destroy_process_group()
    # except Exception as e:
    #     print(f"   Error during conceptual distributed test: {e}")


