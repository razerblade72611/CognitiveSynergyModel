# cognitive_synergy/training/optimizers.py
"""
Optimizer and Learning Rate Scheduler Configuration.

Provides helper functions to create optimizers (e.g., AdamW) and
learning rate schedulers based on configuration parameters.
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.optimizer import Optimizer
# Use specific scheduler base class for better type hinting if possible
# from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Iterable, Optional, List, Tuple, Union

# ==============================================================================
# Optimizer Creation
# ==============================================================================

def create_optimizer(
    model_parameters: Iterable[torch.nn.Parameter],
    config: Dict
) -> Optimizer:
    """
    Creates an optimizer based on the provided configuration.

    Args:
        model_parameters (Iterable[torch.nn.Parameter]): Parameters of the model to optimize.
                                                         Typically `model.parameters()`.
        config (Dict): Configuration dictionary, expected to contain optimizer settings
                       under a key like 'optimizer' or directly within 'training'.
                       Example keys: 'optimizer' (type), 'learning_rate', 'weight_decay',
                       'betas', 'eps', 'momentum'.

    Returns:
        Optimizer: The instantiated PyTorch optimizer.
    """
    # Extract relevant sub-config or use top-level keys
    # Allows flexibility if optimizer settings are nested or at the root of the passed config
    opt_config = config.get('optimizer_settings', config)

    optimizer_type = opt_config.get('optimizer', 'adamw').lower() # Default to adamw
    learning_rate = opt_config.get('learning_rate', 1e-4)
    weight_decay = opt_config.get('weight_decay', 0.01)

    print(f"Creating optimizer: type={optimizer_type}, lr={learning_rate}, weight_decay={weight_decay}")

    # Basic validation
    if learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got {learning_rate}")
    if weight_decay < 0:
        raise ValueError(f"Weight decay cannot be negative, got {weight_decay}")

    # Filter parameters that require gradients - essential for optimization
    # Convert iterable to list to easily check if it's empty and count elements
    params_to_optimize = list(filter(lambda p: p.requires_grad, model_parameters))
    if not params_to_optimize:
        # This can happen if model parts are frozen or if called before model init
        raise ValueError("No model parameters found that require gradients. Check model freezing or initialization order.")
    print(f"  Optimizing {len(params_to_optimize)} parameter groups/tensors.")

    optimizer: Optimizer # Type hint

    if optimizer_type == 'adamw':
        # AdamW is generally preferred for transformer models due to better weight decay handling
        betas = opt_config.get('betas', (0.9, 0.999)) # Default AdamW betas
        eps = opt_config.get('eps', 1e-8)        # Default AdamW epsilon
        if not (isinstance(betas, (list, tuple)) and len(betas) == 2 and all(isinstance(b, float) for b in betas)):
            raise TypeError(f"AdamW 'betas' must be a tuple/list of two floats, got {betas}")
        optimizer = optim.AdamW(
            params=params_to_optimize,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        print(f"  Using AdamW with betas={betas}, eps={eps}")

    elif optimizer_type == 'adam':
        # Standard Adam optimizer
        betas = opt_config.get('betas', (0.9, 0.999)) # Default Adam betas
        eps = opt_config.get('eps', 1e-8)        # Default Adam epsilon
        if not (isinstance(betas, (list, tuple)) and len(betas) == 2 and all(isinstance(b, float) for b in betas)):
            raise TypeError(f"Adam 'betas' must be a tuple/list of two floats, got {betas}")
        optimizer = optim.Adam(
            params=params_to_optimize,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay # Adam does handle weight decay, though differently than AdamW
        )
        print(f"  Using Adam with betas={betas}, eps={eps}")

    elif optimizer_type == 'sgd':
        # Stochastic Gradient Descent
        momentum = opt_config.get('momentum', 0.9) # Common momentum value
        nesterov = opt_config.get('nesterov', False) # Whether to use Nesterov momentum
        if not (0.0 <= momentum <= 1.0):
             raise ValueError(f"Momentum must be between 0.0 and 1.0, got {momentum}")
        optimizer = optim.SGD(
            params=params_to_optimize,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        print(f"  Using SGD with momentum={momentum}, nesterov={nesterov}")

    else:
        raise ValueError(f"Unsupported optimizer type: '{optimizer_type}'. Choose 'adamw', 'adam', or 'sgd'.")

    print("Optimizer created successfully.")
    return optimizer

# ==============================================================================
# Learning Rate Scheduler Creation
# ==============================================================================

def create_scheduler(
    optimizer: Optimizer,
    config: Dict,
    steps_per_epoch: Optional[int] = None # Needed for some schedulers like OneCycleLR or step-based cosine
) -> Optional[lr_scheduler._LRScheduler]: # Use specific base class from PyTorch 1.11+
    """
    Creates a learning rate scheduler based on the provided configuration.

    Args:
        optimizer (Optimizer): The optimizer whose learning rate will be scheduled.
        config (Dict): Configuration dictionary, expected to contain scheduler settings
                       under a key like 'scheduler_settings' or directly within 'training'.
                       Example keys: 'scheduler' (type), 'epochs', 'warmup_epochs', 'min_lr',
                       'step_size', 'milestones', 'decay_rate'.
        steps_per_epoch (Optional[int]): Total number of training steps in one epoch.
                                         Required for schedulers like 'cosine' with warmup
                                         or 'onecycle', or step-based schedulers.

    Returns:
        Optional[lr_scheduler.LRScheduler]: The instantiated PyTorch LR scheduler, or None if no scheduler is configured.
    """
    # Extract relevant sub-config or use top-level keys
    sched_config = config.get('scheduler_settings', config) # Look for nested or use main config

    scheduler_type = sched_config.get('scheduler', None) # Default to None (no scheduler)

    if not scheduler_type:
        print("No learning rate scheduler configured.")
        return None

    scheduler_type = scheduler_type.lower()
    # Get total epochs from training config if available, needed for some schedulers
    # Fallback to scheduler config if not in training config
    epochs = config.get('training', {}).get('epochs', sched_config.get('epochs', None))

    print(f"Creating LR scheduler: type={scheduler_type}")

    scheduler: Optional[lr_scheduler.LRScheduler] = None # Type hint

    if scheduler_type == 'cosine':
        # Cosine Annealing scheduler
        if epochs is None:
            raise ValueError("Total 'epochs' must be specified in config (e.g., under 'training') for 'cosine' scheduler.")
        min_lr = sched_config.get('min_lr', 0.0) # Minimum learning rate for cosine decay
        warmup_epochs = sched_config.get('warmup_epochs', 0)

        # Validate parameters
        if epochs <= 0: raise ValueError("Total 'epochs' must be positive.")
        if warmup_epochs < 0: raise ValueError("warmup_epochs cannot be negative.")
        if warmup_epochs >= epochs: print(f"Warning: warmup_epochs ({warmup_epochs}) >= total epochs ({epochs}).")
        if min_lr < 0.0: raise ValueError("min_lr cannot be negative.")

        if warmup_epochs > 0:
            # Cosine annealing with linear warmup
            if steps_per_epoch is None or steps_per_epoch <= 0:
                raise ValueError("steps_per_epoch (positive integer) must be provided for 'cosine' scheduler with warmup.")

            warmup_steps = warmup_epochs * steps_per_epoch
            # Total steps for the main cosine decay phase
            main_cosine_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch) # Ensure at least 1 step

            # Base learning rate from optimizer
            base_lr = optimizer.defaults.get('lr', None)
            if base_lr is None:
                 # Fallback to config if not in optimizer defaults (should be rare)
                 base_lr = sched_config.get('learning_rate', 1e-4)
            warmup_lr_init = sched_config.get('warmup_lr_init', min(1e-6, base_lr * 0.01)) # Sensible default starting LR

            # Use SequentialLR to combine Linear Warmup and Cosine Decay
            # Ensure start_factor is calculated correctly relative to base_lr
            start_factor = warmup_lr_init / base_lr if base_lr > 1e-9 else 0.0 # Avoid division by zero/tiny LR
            # LinearLR increases LR from start_factor*base_lr to end_factor*base_lr
            warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps)
            # CosineAnnealingLR decreases LR from the current LR (should be base_lr after warmup) down to eta_min
            cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_cosine_steps, eta_min=min_lr)

            # Milestones indicate the step (not epoch) at which to switch schedulers
            scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

            print(f"  Using Cosine scheduler with {warmup_epochs} warmup epochs ({warmup_steps} steps) from {warmup_lr_init:.2g} to {base_lr:.2g}, "
                  f"then decay over {max(0, epochs - warmup_epochs)} epochs ({main_cosine_steps} steps) to {min_lr:.2g}.")

        else:
            # Cosine annealing without warmup - step per batch/step
            if steps_per_epoch is None or steps_per_epoch <= 0:
                 raise ValueError("steps_per_epoch (positive integer) must be provided for step-based 'cosine' scheduler without warmup.")
            total_steps = epochs * steps_per_epoch
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps, # Total number of steps for one cycle
                eta_min=min_lr     # Minimum learning rate
            )
            print(f"  Using Cosine scheduler decaying over {epochs} epochs ({total_steps} steps) to {min_lr:.2g}.")

    elif scheduler_type == 'step':
        # StepLR decays the learning rate by gamma every step_size epochs.
        step_size = sched_config.get('step_size', None) # Number of epochs between LR decays
        decay_rate = sched_config.get('decay_rate', 0.1) # Multiplicative factor of decay (gamma)
        if step_size is None or not isinstance(step_size, int) or step_size <= 0:
            raise ValueError("'step_size' (in epochs, positive integer) must be specified for 'step' scheduler.")
        if not (0.0 < decay_rate <= 1.0):
             raise ValueError(f"decay_rate (gamma) must be between 0 and 1, got {decay_rate}")
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=step_size, # Step based on epochs
            gamma=decay_rate
        )
        print(f"  Using StepLR scheduler: decay LR by factor {decay_rate} every {step_size} epochs.")

    elif scheduler_type == 'multistep':
        # MultiStepLR decays the learning rate by gamma at specified milestone epochs.
        milestones = sched_config.get('milestones', None) # List of epoch indices where LR decays
        decay_rate = sched_config.get('decay_rate', 0.1) # Multiplicative factor of decay (gamma)
        if milestones is None or not isinstance(milestones, list) or not all(isinstance(m, int) and m > 0 for m in milestones):
             raise ValueError("'milestones' (list of positive epoch indices) must be specified for 'multistep' scheduler.")
        if not (0.0 < decay_rate <= 1.0):
             raise ValueError(f"decay_rate (gamma) must be between 0 and 1, got {decay_rate}")
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sorted(milestones), # Ensure milestones are sorted
            gamma=decay_rate
        )
        print(f"  Using MultiStepLR scheduler: decay LR by factor {decay_rate} at epochs {sorted(milestones)}.")

    # Add other schedulers like ReduceLROnPlateau if needed
    # elif scheduler_type == 'reduce_on_plateau':
    #     mode = sched_config.get('mode', 'min') # 'min' or 'max' based on validation metric
    #     factor = sched_config.get('factor', 0.1) # Factor by which LR is reduced
    #     patience = sched_config.get('patience', 10) # Epochs to wait for improvement
    #     threshold = sched_config.get('threshold', 1e-4) # Threshold for measuring improvement
    #     scheduler = lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
    #     )
    #     print(f"  Using ReduceLROnPlateau scheduler: mode={mode}, factor={factor}, patience={patience}.")
    #     # Note: ReduceLROnPlateau requires passing the metric value to scheduler.step(metric)

    else:
        raise ValueError(f"Unsupported scheduler type: '{scheduler_type}'. Choose from 'cosine', 'step', 'multistep'.") # Add others as implemented

    print("LR Scheduler created successfully.")
    return scheduler

# Example Usage (Conceptual - requires model and config)
if __name__ == "__main__":
    print("\n--- Example Optimizer and Scheduler Creation ---")
    # Dummy model and parameters
    dummy_model = nn.Linear(10, 2)

    # Example Config 1: AdamW + Cosine with Warmup
    config1 = {
        'optimizer': 'adamw',
        'learning_rate': 5e-4,
        'weight_decay': 0.05,
        'scheduler': 'cosine',
        'epochs': 50, # Needs total epochs
        'warmup_epochs': 5,
        'min_lr': 1e-6,
        'warmup_lr_init': 1e-7
    }
    print("\nConfig 1:", config1)
    # Need to get parameters again if they were modified by previous optimizers
    opt1 = create_optimizer(dummy_model.parameters(), config1)
    # Need steps_per_epoch for scheduler if using warmup or step-based cosine
    try:
        sched1 = create_scheduler(opt1, config1, steps_per_epoch=100) # Assume 100 steps/epoch
        print("Optimizer 1:", opt1)
        print("Scheduler 1:", sched1)
        # Test scheduler step
        # print("LR at step 0:", sched1.get_last_lr())
        # for _ in range(500): sched1.step() # Simulate 5 epochs warmup
        # print("LR at step 500 (end of warmup):", sched1.get_last_lr())
        # for _ in range(100): sched1.step() # Simulate 1 epoch cosine
        # print("LR at step 600:", sched1.get_last_lr())

    except ValueError as e:
        print(f"Error creating scheduler 1: {e}")


    # Example Config 2: SGD + StepLR
    config2 = {
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'scheduler': 'step',
        'step_size': 15, # Decay every 15 epochs
        'decay_rate': 0.1
    }
    print("\nConfig 2:", config2)
    # Re-get parameters
    dummy_model_2 = nn.Linear(10, 2)
    opt2 = create_optimizer(dummy_model_2.parameters(), config2)
    try:
        sched2 = create_scheduler(opt2, config2) # steps_per_epoch not needed for StepLR
        print("Optimizer 2:", opt2)
        print("Scheduler 2:", sched2)
        # Test scheduler step (epoch based)
        # print("LR at epoch 0:", sched2.get_last_lr())
        # for _ in range(15): sched2.step()
        # print("LR at epoch 15:", sched2.get_last_lr()) # Should have decayed
    except ValueError as e:
        print(f"Error creating scheduler 2: {e}")


    # Example Config 3: AdamW, no scheduler
    config3 = {
        'optimizer': 'adamw',
        'learning_rate': 1e-3,
        'scheduler': None # Explicitly no scheduler
    }
    print("\nConfig 3:", config3)
    dummy_model_3 = nn.Linear(10, 2)
    opt3 = create_optimizer(dummy_model_3.parameters(), config3)
    try:
        sched3 = create_scheduler(opt3, config3)
        print("Optimizer 3:", opt3)
        print("Scheduler 3:", sched3) # Should be None
    except ValueError as e:
        print(f"Error creating scheduler 3: {e}")


