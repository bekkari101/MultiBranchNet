"""
Hyperparameter configuration for Parallel Branch Neural Network on MNIST.

This file contains all hyperparameters for the ParallelBranchNet architecture
and training configuration. Modify these values to experiment with different
model configurations and training settings.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass
class ModelConfig:
    """Model architecture hyperparameters for ParallelBranchNet."""
    
    # Input configuration
    input_size: int = 784  # 28x28 flattened MNIST images
    input_channels: int = 1  # Grayscale images
    input_height: int = 28
    input_width: int = 28
    num_classes: int = 10  # Digits 0-9
    
    # Convolutional feature extractor
    conv_channels: Tuple[int, int] = (12, 16)  # (first_conv, second_conv)
    conv_kernel_size: int = 3
    conv_padding: int = 1
    pool_kernel_size: int = 2
    pool_stride: int = 2
    
    # Branch configuration
    branches: int = 6  # Number of parallel branches
    branch_size: int = 6  # Hidden size per branch (output of each branch)
    branch_layers: int = 2  # Number of FC layers per branch
    
    # Computed properties (don't modify directly)
    @property
    def flat_size(self) -> int:
        """Size after conv layers and flattening."""
        # After conv: second conv channels, 7x7 spatial (after 2 maxpools)
        return self.conv_channels[1] * 7 * 7
    
    @property
    def input_branch_size(self) -> int:
        """Input size per branch after splitting."""
        padded_size = self.flat_size + self.pad_size
        return padded_size // self.branches
    
    @property
    def pad_size(self) -> int:
        """Padding needed to make flat_size divisible by branches."""
        return (self.branches - self.flat_size % self.branches) % self.branches
    
    @property
    def hidden_size(self) -> int:
        """Total hidden size after concatenating all branches."""
        return self.branches * self.branch_size
    
    def get_trainable_parameters(self) -> int:
        """Calculate total trainable parameters for this architecture."""
        # Conv layers
        conv1_params = self.conv_channels[0] * self.input_channels * self.conv_kernel_size * self.conv_kernel_size
        conv2_params = self.conv_channels[1] * self.conv_channels[0] * self.conv_kernel_size * self.conv_kernel_size
        conv_params = conv1_params + conv2_params
        
        # Branch layers (each branch processes only its chunk)
        input_per_branch = self.input_branch_size
        branch_params = 0
        
        # First layer: input_chunk -> branch_size
        branch_params += self.branches * (input_per_branch * self.branch_size + self.branch_size)
        
        # Middle layers: branch_size -> branch_size (branch_layers - 1 times)
        for _ in range(self.branch_layers - 1):
            branch_params += self.branches * (self.branch_size * self.branch_size + self.branch_size)
        
        # Output layer: all branches -> num_classes
        output_params = self.hidden_size * self.num_classes + self.num_classes
        
        total_params = conv_params + branch_params + output_params
        return total_params


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Data configuration
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training loop
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
    lr_step_size: int = 20  # For step scheduler
    lr_gamma: float = 0.1  # For step scheduler
    lr_patience: int = 10  # For plateau scheduler
    lr_min: float = 1e-6  # Minimum learning rate
    
    # Regularization
    dropout_rate: float = 0.2
    batch_norm: bool = True
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    # Data paths
    data_root: str = "data"
    train_dir: str = "train"
    valid_dir: str = "valid"
    test_dir: str = "test"
    
    # Data splits
    train_size: int = 45000
    valid_size: int = 15000
    test_size: int = 10000
    
    # Data augmentation
    use_augmentation: bool = False  # Disabled as requested
    rotation_degrees: float = 10.0
    translate: Tuple[float, float] = (0.1, 0.1)  # (width, height) as fraction
    scale: Tuple[float, float] = (0.9, 1.1)
    brightness: float = 0.2
    contrast: float = 0.2
    
    # Normalization
    normalize: bool = True
    mean: float = 0.1307  # MNIST mean
    std: float = 0.3081   # MNIST std


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    
    # Logging
    log_interval: int = 100  # Log every N batches
    log_level: str = "INFO"
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    save_last: bool = True
    
    # Visualization
    save_predictions: bool = True
    predictions_dir: str = "predictions"
    num_prediction_samples: int = 100
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs"


@dataclass
class ExperimentConfig:
    """Experiment configuration and reproducibility."""
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    # Mixed precision
    use_amp: bool = False  # Automatic Mixed Precision
    
    # Experiment tracking
    experiment_name: str = "parallel_branch_mnist"
    run_name: Optional[str] = None  # If None, will use timestamp
    
    # Model saving
    save_model: bool = True
    model_save_path: str = "models/parallel_branch_net.pth"


# Default configuration instances
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_LOGGING_CONFIG = LoggingConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()


def get_device(device_config: str) -> torch.device:
    """Get the appropriate device based on configuration."""
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_config)


def print_config_summary():
    """Print a summary of all configuration parameters."""
    print("=" * 60)
    print("PARALLEL BRANCH NETWORK CONFIGURATION")
    print("=" * 60)
    
    print(f"\n📊 MODEL ARCHITECTURE:")
    print(f"  • Input: {DEFAULT_MODEL_CONFIG.input_height}×{DEFAULT_MODEL_CONFIG.input_width} grayscale")
    print(f"  • Branches: {DEFAULT_MODEL_CONFIG.branches}")
    print(f"  • Branch size: {DEFAULT_MODEL_CONFIG.branch_size}")
    print(f"  • Branch layers: {DEFAULT_MODEL_CONFIG.branch_layers}")
    print(f"  • Total hidden size: {DEFAULT_MODEL_CONFIG.hidden_size}")
    print(f"  • Conv channels: {DEFAULT_MODEL_CONFIG.conv_channels}")
    print(f"  • Flat size: {DEFAULT_MODEL_CONFIG.flat_size}")
    print(f"  • Input per branch: {DEFAULT_MODEL_CONFIG.input_branch_size}")
    print(f"  • Padding needed: {DEFAULT_MODEL_CONFIG.pad_size}")
    print(f"  • Trainable parameters: {DEFAULT_MODEL_CONFIG.get_trainable_parameters():,}")
    
    print(f"\n🔗 BRANCH ARCHITECTURE:")
    print(f"  • Each branch processes only {DEFAULT_MODEL_CONFIG.input_branch_size} features")
    print(f"  • Branch 1: features [0:{DEFAULT_MODEL_CONFIG.input_branch_size}]")
    print(f"  • Branch 2: features [{DEFAULT_MODEL_CONFIG.input_branch_size}:{2*DEFAULT_MODEL_CONFIG.input_branch_size}]")
    print(f"  • ... (each branch gets its own chunk)")
    print(f"  • Final layer: all {DEFAULT_MODEL_CONFIG.hidden_size} branch outputs → {DEFAULT_MODEL_CONFIG.num_classes} classes")
    
    print(f"\n🎯 TRAINING:")
    print(f"  • Epochs: {DEFAULT_TRAINING_CONFIG.num_epochs}")
    print(f"  • Batch size: {DEFAULT_TRAINING_CONFIG.batch_size}")
    print(f"  • Learning rate: {DEFAULT_TRAINING_CONFIG.learning_rate}")
    print(f"  • Optimizer: {DEFAULT_TRAINING_CONFIG.optimizer}")
    print(f"  • Weight decay: {DEFAULT_TRAINING_CONFIG.weight_decay}")
    
    print(f"\n📁 DATA:")
    print(f"  • Train: {DEFAULT_DATA_CONFIG.train_size:,} samples")
    print(f"  • Valid: {DEFAULT_DATA_CONFIG.valid_size:,} samples")
    print(f"  • Test: {DEFAULT_DATA_CONFIG.test_size:,} samples")
    print(f"  • Augmentation: {'Yes' if DEFAULT_DATA_CONFIG.use_augmentation else 'No'} (DISABLED)")
    
    print(f"\n🔧 EXPERIMENT:")
    print(f"  • Device: {get_device(DEFAULT_EXPERIMENT_CONFIG.device)}")
    print(f"  • Seed: {DEFAULT_EXPERIMENT_CONFIG.seed}")
    print(f"  • Mixed precision: {'Yes' if DEFAULT_EXPERIMENT_CONFIG.use_amp else 'No'}")
    print(f"  • Experiment name: {DEFAULT_EXPERIMENT_CONFIG.experiment_name}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Print configuration summary when run directly
    print_config_summary()
