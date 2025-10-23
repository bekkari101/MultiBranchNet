"""
Hyperparameter configuration for Parallel Branch Neural Network on MNIST.

This file contains all hyperparameters for the ParallelBranchNet architecture
and training configuration with STRUCTURED CONNECTIVITY support.

CURRENT DEFAULT CONFIGURATION (V2, best @epoch 47: 98.57% Test Accuracy):
=========================================================================
Architecture:
  - Conv Layers: 2 layers [18, 11] with 3x3 kernels
  - Branch Layers: 2 layers with structured connectivity
    * Layer 1: 6 branches × 7 nodes = 42 total
    * Layer 2: 3 branches × 14 nodes = 42 total
    * Connectivity: 6→3 (every 2 branches from L1 connect to 1 branch in L2)
  - Total Parameters: ~6,855

Training:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Batch Size: 128
  - Epochs: 50

Performance (latest run):
  - Best Test Accuracy: 98.57% (epoch 47 snapshot)
  - Best Test F1 Score: 0.9857 (epoch 47)

Features:
  - Extended training automatically triggered when test accuracy reaches 98%
  - Learning rate reduction during extended training for fine-tuning
  - Two-layer branch structure with structured connectivity (6→3)
  - Balanced complexity: more expressive than single-layer, efficient parameters

NEW FEATURES:
- Structured connectivity between branch layers
- Per-layer branch configuration (flexible architecture)
- Multiple optimizer options (Adam, SGD, AdamW, RMSprop)
- Automatic connectivity validation
- Extended training for high-performing models
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import torch


@dataclass
class BranchLayerConfig:
    """Configuration for a single branch layer with structured connectivity."""
    num_branches: int  # Number of parallel branches in this layer
    nodes_per_branch: int  # Number of nodes/neurons in each branch
    
    @property
    def total_nodes(self) -> int:
        """Total output size after concatenating all branches."""
        return self.num_branches * self.nodes_per_branch


@dataclass
class ModelConfig:
    """Model architecture hyperparameters for ParallelBranchNet with structured connectivity."""
    
    # Input configuration
    input_size: int = 784  # 28x28 flattened MNIST images
    input_channels: int = 1  # Grayscale images
    input_height: int = 28
    input_width: int = 28
    num_classes: int = 10  # Digits 0-9
    

    conv_channels: Tuple[int, int] = (18, 11)  # Two conv layers as per configuration
    conv_kernel_size: int = 3
    conv_padding: int = 1
    pool_kernel_size: int = 2
    pool_stride: int = 2
    num_conv_layers: int = 2  # Number of conv layers
    

    branch_layers: List[BranchLayerConfig] = field(default_factory=lambda: [
        BranchLayerConfig(num_branches=6, nodes_per_branch=7),   # Layer 1: 6 branches × 7 nodes
        BranchLayerConfig(num_branches=3, nodes_per_branch=14),  # Layer 2: 3 branches × 14 nodes
    ])
    
    # Computed properties (don't modify directly)
    @property
    def flat_size(self) -> int:
        """Size after conv layers and flattening."""
        # Calculate spatial size after pooling layers
        spatial_size = self.input_height
        for _ in range(self.num_conv_layers):
            spatial_size = spatial_size // 2
        # After conv layers: last conv channels × spatial_size × spatial_size
        return self.conv_channels[-1] * spatial_size * spatial_size
    
    @property
    def num_branch_layers(self) -> int:
        """Total number of branch layers."""
        return len(self.branch_layers)
    
    @property
    def first_layer_branches(self) -> int:
        """Number of branches in the first layer."""
        return self.branch_layers[0].num_branches if self.branch_layers else 0
    
    @property
    def pad_size(self) -> int:
        """Padding needed to make flat_size divisible by first layer branches."""
        if not self.branch_layers:
            return 0
        return (self.first_layer_branches - self.flat_size % self.first_layer_branches) % self.first_layer_branches
    
    @property
    def input_branch_size(self) -> int:
        """Input size per branch after splitting in the first layer."""
        if not self.branch_layers:
            return self.flat_size
        padded_size = self.flat_size + self.pad_size
        return padded_size // self.first_layer_branches
    
    @property
    def final_hidden_size(self) -> int:
        """Total hidden size after the last branch layer."""
        if not self.branch_layers:
            return self.flat_size
        return self.branch_layers[-1].total_nodes
    
    def validate_structured_connectivity(self) -> Tuple[bool, List[str]]:
        """
        Validate that the branch configuration supports structured connectivity.
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        if len(self.branch_layers) < 2:
            return True, []  # Single layer doesn't need validation
        
        for i in range(len(self.branch_layers) - 1):
            current_layer = self.branch_layers[i]
            next_layer = self.branch_layers[i + 1]
            
            # Check if current layer branches are evenly divisible by next layer branches
            if current_layer.num_branches % next_layer.num_branches != 0:
                issues.append(
                    f"Layer {i+1}→{i+2}: {current_layer.num_branches} branches cannot be "
                    f"evenly divided into {next_layer.num_branches} branches. "
                    f"Ratio: {current_layer.num_branches}/{next_layer.num_branches} = "
                    f"{current_layer.num_branches / next_layer.num_branches:.2f} (should be integer)"
                )
            else:
                branches_per_group = current_layer.num_branches // next_layer.num_branches
                # This is valid - log the connectivity pattern
                print(f"✓ Layer {i+1}→{i+2}: Every {branches_per_group} branches from layer {i+1} "
                      f"connect to 1 branch in layer {i+2}")
        
        return len(issues) == 0, issues
    
    def get_connectivity_pattern(self) -> List[dict]:
        """
        Get the structured connectivity pattern between layers.
        Returns a list of connectivity information for each layer transition.
        """
        if len(self.branch_layers) < 2:
            return []
        
        connectivity = []
        
        for i in range(len(self.branch_layers) - 1):
            current_layer = self.branch_layers[i]
            next_layer = self.branch_layers[i + 1]
            
            if current_layer.num_branches % next_layer.num_branches == 0:
                branches_per_group = current_layer.num_branches // next_layer.num_branches
                
                # Build the connection mapping
                connections = []
                for next_branch_idx in range(next_layer.num_branches):
                    source_branches = list(range(
                        next_branch_idx * branches_per_group,
                        (next_branch_idx + 1) * branches_per_group
                    ))
                    connections.append({
                        'target_branch': next_branch_idx,
                        'source_branches': source_branches,
                        'num_inputs': branches_per_group * current_layer.nodes_per_branch
                    })
                
                connectivity.append({
                    'from_layer': i + 1,
                    'to_layer': i + 2,
                    'branches_per_group': branches_per_group,
                    'connections': connections
                })
            else:
                connectivity.append({
                    'from_layer': i + 1,
                    'to_layer': i + 2,
                    'branches_per_group': None,
                    'connections': None,
                    'error': 'Invalid: branches not evenly divisible'
                })
        
        return connectivity
    
    def get_trainable_parameters(self) -> int:
        """
        Calculate total trainable parameters for this architecture.
        Fixed to match actual PyTorch model parameter count.
        """
        # Conv layers - support variable number of layers
        # NOTE: The model implementation uses 2 convolutional layers (conv1, conv2).
        conv_params = 0
        prev_channels = self.input_channels
        for i in range(2):  # Hardcoded to 2 layers to match model implementation
            curr_channels = self.conv_channels[i]
            # Weights: in_channels × out_channels × kernel_h × kernel_w
            # Bias: out_channels
            weight_params = prev_channels * curr_channels * self.conv_kernel_size * self.conv_kernel_size
            bias_params = curr_channels
            conv_params += weight_params + bias_params
            prev_channels = curr_channels
        
        # Branch layers parameters with structured connectivity
        branch_params = 0
        
        for layer_idx, layer_config in enumerate(self.branch_layers):
            if layer_idx == 0:
                # First layer: split the flattened conv output
                padded_size = self.flat_size + self.pad_size
                input_per_branch = padded_size // layer_config.num_branches
            else:
                # Subsequent layers: structured connectivity from previous layer
                prev_layer_config = self.branch_layers[layer_idx - 1]
                
                # Check if structured connectivity is valid
                if prev_layer_config.num_branches % layer_config.num_branches == 0:
                    # Each branch in current layer receives from multiple branches in previous layer
                    branches_per_group = prev_layer_config.num_branches // layer_config.num_branches
                    input_per_branch = branches_per_group * prev_layer_config.nodes_per_branch
                else:
                    # Fallback: full connectivity (not recommended)
                    input_per_branch = prev_layer_config.total_nodes // layer_config.num_branches
            
            # Each branch has its own Linear layer: (input_per_branch * nodes_per_branch) weights + nodes_per_branch biases
            # Total for all branches: num_branches × (input_per_branch × nodes_per_branch + nodes_per_branch)
            layer_params = layer_config.num_branches * (
                input_per_branch * layer_config.nodes_per_branch + layer_config.nodes_per_branch
            )
            branch_params += layer_params
        
        # Output layer: Linear(final_hidden_size, num_classes)
        # Weights: final_hidden_size × num_classes
        # Bias: num_classes
        output_params = self.final_hidden_size * self.num_classes + self.num_classes
        
        total_params = conv_params + branch_params + output_params
        return total_params
    
    def get_layer_info(self) -> List[dict]:
        """Get detailed information about each branch layer with structured connectivity."""
        layer_info = []
        
        for layer_idx, layer_config in enumerate(self.branch_layers):
            if layer_idx == 0:
                # First layer
                padded_size = self.flat_size + self.pad_size
                input_per_branch = padded_size // layer_config.num_branches
                total_input = padded_size
                connectivity_type = "Split from Conv Output"
            else:
                # Subsequent layers with structured connectivity
                prev_layer_config = self.branch_layers[layer_idx - 1]
                
                if prev_layer_config.num_branches % layer_config.num_branches == 0:
                    branches_per_group = prev_layer_config.num_branches // layer_config.num_branches
                    input_per_branch = branches_per_group * prev_layer_config.nodes_per_branch
                    connectivity_type = f"Structured: {branches_per_group} branches → 1 branch"
                else:
                    input_per_branch = prev_layer_config.total_nodes // layer_config.num_branches
                    connectivity_type = "Warning: Full connectivity (not evenly divisible)"
                
                total_input = prev_layer_config.total_nodes
            
            layer_info.append({
                'layer_idx': layer_idx + 1,
                'num_branches': layer_config.num_branches,
                'nodes_per_branch': layer_config.nodes_per_branch,
                'input_per_branch': input_per_branch,
                'total_input': total_input,
                'total_output': layer_config.total_nodes,
                'connectivity_type': connectivity_type,
            })
        
        return layer_info
    
    def get_actual_parameters(self) -> int:
        """
        Build the actual model and count real trainable parameters.
        This is more accurate than the estimated count.
        
        Returns:
            int: Actual number of trainable parameters in the model
        """
        try:
            # Import here to avoid circular dependency
            from model import ParallelBranchNet
            
            # Create temporary model
            model = ParallelBranchNet(self)
            
            # Count actual trainable parameters
            actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Clean up
            del model
            
            return actual_params
        except Exception as e:
            print(f"⚠️ Could not build model to count actual parameters: {e}")
            return -1
    
    def verify_parameter_count(self, verbose: bool = True) -> dict:
        """
        Verify parameter count by comparing estimated vs actual.
        
        Args:
            verbose: If True, print detailed comparison
            
        Returns:
            dict: Dictionary with estimated, actual, difference, and match status
        """
        estimated = self.get_trainable_parameters()
        actual = self.get_actual_parameters()
        
        if actual == -1:
            return {
                'estimated': estimated,
                'actual': None,
                'difference': None,
                'match': False,
                'error': 'Could not build model'
            }
        
        difference = actual - estimated
        match = (difference == 0)
        
        if verbose:
            print("\n" + "=" * 80)
            print("PARAMETER COUNT VERIFICATION")
            print("=" * 80)
            print(f"📊 Estimated parameters: {estimated:,}")
            print(f"🔍 Actual parameters:    {actual:,}")
            print(f"📈 Difference:           {difference:+,}")
            
            if match:
                print("✅ MATCH! Estimation is accurate.")
            else:
                print("⚠️  MISMATCH! Estimation differs from actual count.")
                percent_diff = (abs(difference) / actual) * 100
                print(f"   Difference: {percent_diff:.2f}%")
            print("=" * 80)
        
        return {
            'estimated': estimated,
            'actual': actual,
            'difference': difference,
            'match': match,
            'percent_difference': abs(difference) / actual * 100 if actual > 0 else 0
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters - current default configuration."""
    
    # Data configuration
    batch_size: int = 128  # Best model used 256
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training loop - BEST MODEL CONFIG
    num_epochs: int = 50  # Slightly longer training
    learning_rate: float = 0.001  # Adjusted per provided configuration
    weight_decay: float = 5e-5  # Slightly increased regularization
    
    # Optimizer - BEST MODEL: ADAM
    optimizer: str = "adam"  # Adam is best for MNIST
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Learning rate scheduling - DISABLED for constant LR
    lr_scheduler: str = "plateau"  # Use ReduceLROnPlateau for fine-tuning
    lr_patience: int = 12  # Wait longer before reducing LR
    lr_gamma: float = 0.5  # Reduce LR less aggressively
    lr_min: float = 1e-4  # Don't let LR get too low
    
    # Regularization
    dropout_rate: float = 0.0
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
    use_augmentation: bool = False  # Disabled by default
    rotation_degrees: float = 10.0
    translate: Tuple[float, float] = (0.1, 0.1)  # (width, height) as fraction
    scale: Tuple[float, float] = (0.9, 1.1)
    brightness: float = 0.2
    contrast: float = 0.2
    
    # Normalization
    normalize: bool = False
    mean: float = 0.1307  # MNIST mean
    std: float = 0.3081   # MNIST std


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    
    # Logging
    log_interval: int = 50  # Log every N batches
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
    print("=" * 80)
    print("PARALLEL BRANCH NETWORK CONFIGURATION - STRUCTURED CONNECTIVITY")
    print("=" * 80)
    
    # Validate connectivity first
    is_valid, issues = DEFAULT_MODEL_CONFIG.validate_structured_connectivity()
    
    if not is_valid:
        print("\n⚠️  CONNECTIVITY VALIDATION ISSUES:")
        for issue in issues:
            print(f"  ❌ {issue}")
        print()
    
    print(f"\n📊 MODEL ARCHITECTURE:")
    print(f"  • Input: {DEFAULT_MODEL_CONFIG.input_height}×{DEFAULT_MODEL_CONFIG.input_width} grayscale")
    print(f"  • Conv channels: {DEFAULT_MODEL_CONFIG.conv_channels}")
    print(f"  • Flat size: {DEFAULT_MODEL_CONFIG.flat_size}")
    print(f"  • Padding needed: {DEFAULT_MODEL_CONFIG.pad_size}")
    print(f"  • Total branch layers: {DEFAULT_MODEL_CONFIG.num_branch_layers}")
    
    # Verify parameter count with actual model
    param_verification = DEFAULT_MODEL_CONFIG.verify_parameter_count(verbose=False)
    if param_verification['actual'] is not None:
        if param_verification['match']:
            print(f"  • Trainable parameters: {param_verification['actual']:,} ✅")
        else:
            print(f"  • Estimated parameters: {param_verification['estimated']:,}")
            print(f"  • Actual parameters: {param_verification['actual']:,} (diff: {param_verification['difference']:+,})")
    else:
        print(f"  • Trainable parameters (estimated): {DEFAULT_MODEL_CONFIG.get_trainable_parameters():,}")
    
    print(f"\n🔗 BRANCH LAYER ARCHITECTURE (STRUCTURED CONNECTIVITY):")
    layer_info = DEFAULT_MODEL_CONFIG.get_layer_info()
    for info in layer_info:
        print(f"\n  Layer {info['layer_idx']}:")
        print(f"    • Branches: {info['num_branches']}")
        print(f"    • Nodes per branch: {info['nodes_per_branch']}")
        print(f"    • Input per branch: {info['input_per_branch']}")
        print(f"    • Total input: {info['total_input']}")
        print(f"    • Total output: {info['total_output']}")
        print(f"    • Connectivity: {info['connectivity_type']}")
    
    print(f"\n  Final Layer (Classification):")
    print(f"    • Input: {DEFAULT_MODEL_CONFIG.final_hidden_size}")
    print(f"    • Output: {DEFAULT_MODEL_CONFIG.num_classes} classes")
    
    # Print detailed connectivity pattern
    print(f"\n🔀 DETAILED CONNECTIVITY PATTERN:")
    connectivity = DEFAULT_MODEL_CONFIG.get_connectivity_pattern()
    for conn in connectivity:
        if conn.get('error'):
            print(f"\n  ❌ Layer {conn['from_layer']} → Layer {conn['to_layer']}: {conn['error']}")
        else:
            print(f"\n  Layer {conn['from_layer']} → Layer {conn['to_layer']}:")
            print(f"    • Pattern: Every {conn['branches_per_group']} branches → 1 branch")
            for c in conn['connections']:
                print(f"    • Branch {c['target_branch']} ← Branches {c['source_branches']} "
                      f"({c['num_inputs']} inputs)")
    
    print(f"\n🎯 TRAINING:")
    print(f"  • Epochs: {DEFAULT_TRAINING_CONFIG.num_epochs}")
    print(f"  • Batch size: {DEFAULT_TRAINING_CONFIG.batch_size}")
    print(f"  • Learning rate: {DEFAULT_TRAINING_CONFIG.learning_rate}")
    print(f"  • Optimizer: {DEFAULT_TRAINING_CONFIG.optimizer.upper()}")
    print(f"  • Weight decay: {DEFAULT_TRAINING_CONFIG.weight_decay}")
    
    print(f"\n📁 DATA:")
    print(f"  • Train: {DEFAULT_DATA_CONFIG.train_size:,} samples")
    print(f"  • Valid: {DEFAULT_DATA_CONFIG.valid_size:,} samples")
    print(f"  • Test: {DEFAULT_DATA_CONFIG.test_size:,} samples")
    print(f"  • Augmentation: {'Yes' if DEFAULT_DATA_CONFIG.use_augmentation else 'No'}")
    
    print(f"\n🔧 EXPERIMENT:")
    print(f"  • Device: {get_device(DEFAULT_EXPERIMENT_CONFIG.device)}")
    print(f"  • Seed: {DEFAULT_EXPERIMENT_CONFIG.seed}")
    print(f"  • Mixed precision: {'Yes' if DEFAULT_EXPERIMENT_CONFIG.use_amp else 'No'}")
    print(f"  • Experiment name: {DEFAULT_EXPERIMENT_CONFIG.experiment_name}")
    
    print("=" * 80)


# Example configurations with structured connectivity
def create_8_to_4_to_2_config() -> ModelConfig:
    """Create an 8→4→2 structured configuration."""
    return ModelConfig(
        branch_layers=[
            BranchLayerConfig(num_branches=8, nodes_per_branch=8),  # 64 total
            BranchLayerConfig(num_branches=4, nodes_per_branch=6),  # 24 total (2:1 ratio)
            BranchLayerConfig(num_branches=2, nodes_per_branch=4),  # 8 total (2:1 ratio)
        ]
    )


def create_6_to_3_config() -> ModelConfig:
    """Create a 6→3 structured configuration."""
    return ModelConfig(
        branch_layers=[
            BranchLayerConfig(num_branches=6, nodes_per_branch=8),  # 48 total
            BranchLayerConfig(num_branches=3, nodes_per_branch=6),  # 18 total (2:1 ratio)
        ]
    )


def create_simple_config() -> ModelConfig:
    """Create a simple single-layer configuration."""
    return ModelConfig(
        branch_layers=[
            BranchLayerConfig(num_branches=4, nodes_per_branch=8),  # 32 total
        ]
    )


if __name__ == "__main__":
    # Print default configuration
    print_config_summary()
    
    print("\n\n" + "=" * 80)
    print("EXAMPLE CONFIGURATIONS")
    print("=" * 80)
    
    # Example 1: 8→4→2
    print("\n1. 8→4→2 Configuration:")
    config1 = create_8_to_4_to_2_config()
    is_valid, issues = config1.validate_structured_connectivity()
    print(f"   Valid: {'✓ Yes' if is_valid else '✗ No'}")
    
    # Example 2: 6→3
    print("\n2. 6→3 Configuration:")
    config2 = create_6_to_3_config()
    is_valid, issues = config2.validate_structured_connectivity()
    print(f"   Valid: {'✓ Yes' if is_valid else '✗ No'}")
    
    # Verify parameter count for default configuration
    print("\n\n" + "=" * 80)
    print("PARAMETER COUNT VERIFICATION (Default Configuration)")
    print("=" * 80)
    verification = DEFAULT_MODEL_CONFIG.verify_parameter_count(verbose=True)