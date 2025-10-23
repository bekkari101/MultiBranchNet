"""
Parallel Branch Network with STRUCTURED CONNECTIVITY.

This implementation creates a neural network where branches from one layer
connect to the next layer in structured groups, rather than fully connected.

Example: 8 branches → 4 branches
- Branches [0,1] connect to Branch 0
- Branches [2,3] connect to Branch 1
- Branches [4,5] connect to Branch 2
- Branches [6,7] connect to Branch 3
"""

import torch
import torch.nn as nn
from typing import List
from hyperP import ModelConfig


class StructuredBranchLayer(nn.Module):
    """
    A single layer with structured branch connectivity.
    
    Each branch in this layer receives inputs from a specific group of branches
    from the previous layer (not all branches).
    """
    
    def __init__(
        self, 
        num_branches: int,
        nodes_per_branch: int,
        input_per_branch: int,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.num_branches = num_branches
        self.nodes_per_branch = nodes_per_branch
        self.input_per_branch = input_per_branch
        
        # Create separate linear layer for each branch
        # Each branch only processes its designated input chunk
        self.branches = nn.ModuleList([
            nn.Linear(input_per_branch, nodes_per_branch)
            for _ in range(num_branches)
        ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, connectivity_info: dict = None) -> torch.Tensor:
        """
        Forward pass with structured connectivity.
        
        Args:
            x: Input tensor of shape (batch_size, total_input_size)
            connectivity_info: Dictionary containing connection mappings
        
        Returns:
            Concatenated output from all branches
        """
        batch_size = x.size(0)
        branch_outputs = []
        
        if connectivity_info is None:
            # Simple split for first layer (no structured connectivity needed)
            branch_inputs = torch.split(x, self.input_per_branch, dim=1)
            for branch_idx, branch_linear in enumerate(self.branches):
                h = self.relu(branch_linear(branch_inputs[branch_idx]))
                h = self.dropout(h)
                branch_outputs.append(h)
        else:
            # Structured connectivity: each branch gets specific inputs
            connections = connectivity_info['connections']
            
            for target_branch_idx in range(self.num_branches):
                # Get source branches for this target
                source_branch_indices = connections[target_branch_idx]['source_branches']
                nodes_per_source = connections[target_branch_idx]['num_inputs'] // len(source_branch_indices)
                
                # Extract outputs from source branches
                branch_inputs = []
                for source_idx in source_branch_indices:
                    start = source_idx * nodes_per_source
                    end = (source_idx + 1) * nodes_per_source
                    branch_inputs.append(x[:, start:end])
                
                # Concatenate inputs from source branches
                combined_input = torch.cat(branch_inputs, dim=1)
                
                # Process through this branch's linear layer
                h = self.relu(self.branches[target_branch_idx](combined_input))
                h = self.dropout(h)
                branch_outputs.append(h)
        
        # Concatenate all branch outputs
        return torch.cat(branch_outputs, dim=1)


class ParallelBranchNet(nn.Module):
    """
    Parallel Branch Network with Structured Connectivity.
    
    Architecture:
    1. Conv layers for feature extraction
    2. Multiple branch layers with structured connectivity
    3. Final classification layer
    
    Key Feature: Branches connect in structured groups between layers
    """
    
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.config = model_config
        
        # Validate structured connectivity
        is_valid, issues = model_config.validate_structured_connectivity()
        if not is_valid:
            raise ValueError(f"Invalid branch configuration:\n" + "\n".join(issues))
        
        # Get connectivity pattern
        self.connectivity_pattern = model_config.get_connectivity_pattern()
        
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(
            self.config.input_channels, 
            self.config.conv_channels[0], 
            kernel_size=self.config.conv_kernel_size,
            padding=self.config.conv_padding
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=self.config.pool_kernel_size,
            stride=self.config.pool_stride
        )
        
        self.conv2 = nn.Conv2d(
            self.config.conv_channels[0],
            self.config.conv_channels[1],
            kernel_size=self.config.conv_kernel_size,
            padding=self.config.conv_padding
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=self.config.pool_kernel_size,
            stride=self.config.pool_stride
        )
        
        # Build structured branch layers
        self.branch_layers = nn.ModuleList()
        
        for layer_idx, layer_config in enumerate(self.config.branch_layers):
            if layer_idx == 0:
                # First layer: split from flattened conv output
                padded_size = self.config.flat_size + self.config.pad_size
                input_per_branch = padded_size // layer_config.num_branches
            else:
                # Subsequent layers: structured connectivity
                prev_layer_config = self.config.branch_layers[layer_idx - 1]
                branches_per_group = prev_layer_config.num_branches // layer_config.num_branches
                input_per_branch = branches_per_group * prev_layer_config.nodes_per_branch
            
            branch_layer = StructuredBranchLayer(
                num_branches=layer_config.num_branches,
                nodes_per_branch=layer_config.nodes_per_branch,
                input_per_branch=input_per_branch,
                dropout_rate=0.0            )
            self.branch_layers.append(branch_layer)
        
        # Output layer
        self.fc_out = nn.Linear(self.config.final_hidden_size, self.config.num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
        print(f"\n🔗 Model initialized with structured connectivity:")
        for conn in self.connectivity_pattern:
            if not conn.get('error'):
                print(f"  Layer {conn['from_layer']}→{conn['to_layer']}: "
                      f"{conn['branches_per_group']} branches → 1 branch")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with structured connectivity.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Reshape if input is flattened
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        # Convolutional feature extraction
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Padding for even split
        if self.config.pad_size > 0:
            x = torch.nn.functional.pad(x, (0, self.config.pad_size))
        
        # Process through structured branch layers
        for layer_idx, branch_layer in enumerate(self.branch_layers):
            if layer_idx == 0:
                # First layer: simple split, no structured connectivity needed
                x = branch_layer(x, connectivity_info=None)
            else:
                # Subsequent layers: apply structured connectivity
                conn_info = self.connectivity_pattern[layer_idx - 1]
                
                if conn_info.get('error'):
                    raise RuntimeError(f"Cannot process layer {layer_idx}: {conn_info['error']}")
                
                x = branch_layer(x, connectivity_info=conn_info)
        
        # Final classification
        x = self.fc_out(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_architecture(self):
        """Print detailed architecture information."""
        print("\n" + "=" * 80)
        print("PARALLEL BRANCH NETWORK ARCHITECTURE - STRUCTURED CONNECTIVITY")
        print("=" * 80)
        
        print("\n📊 CONVOLUTIONAL LAYERS:")
        print(f"  Conv1: {self.config.input_channels} → {self.config.conv_channels[0]} "
              f"(kernel={self.config.conv_kernel_size}x{self.config.conv_kernel_size})")
        print(f"  Pool1: MaxPool (kernel={self.config.pool_kernel_size}x{self.config.pool_kernel_size})")
        print(f"  Conv2: {self.config.conv_channels[0]} → {self.config.conv_channels[1]} "
              f"(kernel={self.config.conv_kernel_size}x{self.config.conv_kernel_size})")
        print(f"  Pool2: MaxPool (kernel={self.config.pool_kernel_size}x{self.config.pool_kernel_size})")
        print(f"  Flattened size: {self.config.flat_size}")
        
        print("\n🔗 STRUCTURED BRANCH LAYERS:")
        layer_info = self.config.get_layer_info()
        for info in layer_info:
            print(f"\n  Layer {info['layer_idx']}:")
            print(f"    • Branches: {info['num_branches']}")
            print(f"    • Nodes per branch: {info['nodes_per_branch']}")
            print(f"    • Input per branch: {info['input_per_branch']}")
            print(f"    • Total output: {info['total_output']}")
            print(f"    • Connectivity: {info['connectivity_type']}")
        
        print("\n🔀 CONNECTIVITY PATTERN:")
        for conn in self.connectivity_pattern:
            if not conn.get('error'):
                print(f"\n  Layer {conn['from_layer']} → Layer {conn['to_layer']}:")
                print(f"    Pattern: Every {conn['branches_per_group']} branches → 1 branch")
                for c in conn['connections'][:2]:  # Show first 2 connections as example
                    print(f"    • Branch {c['target_branch']} ← Branches {c['source_branches']}")
        
        print(f"\n📈 OUTPUT LAYER:")
        print(f"  Input: {self.config.final_hidden_size}")
        print(f"  Output: {self.config.num_classes} classes")
        
        print(f"\n🔢 PARAMETERS:")
        print(f"  Estimated: {self.config.get_trainable_parameters():,}")
        print(f"  Actual: {self.get_num_parameters():,}")
        
        print("=" * 80)


def test_model():
    """Test the structured connectivity model."""
    from hyperP import ModelConfig, BranchLayerConfig
    
    print("Testing Structured Parallel Branch Network\n")
    
    # Create 8→4 configuration
    config = ModelConfig(
        branch_layers=[
            BranchLayerConfig(num_branches=8, nodes_per_branch=4),  # 32 total
            BranchLayerConfig(num_branches=4, nodes_per_branch=2),  # 8 total
        ]
    )
    
    # Create model
    model = ParallelBranchNet(config)
    model.print_architecture()
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {config.num_classes})")
    
    assert output.shape == (batch_size, config.num_classes), "Output shape mismatch!"
    print("✅ Forward pass successful!")
    
    # Test with flattened input
    print("\n🧪 Testing with flattened input...")
    x_flat = torch.randn(batch_size, 784)
    output_flat = model(x_flat)
    print(f"Output shape: {output_flat.shape}")
    assert output_flat.shape == (batch_size, config.num_classes), "Output shape mismatch!"
    print("✅ Flattened input test successful!")


if __name__ == "__main__":
    test_model()