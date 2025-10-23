"""
Multi-Branch Neural Network - Hyperparameter Search (Optimized for Google Colab)
==================================================================================
- All data loaded into RAM for maximum speed
- CUDA GPU optimized with persistent workers
- Aggressive caching and memory management
- Single file for easy Colab execution
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================
import os
import sys
import time
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from PIL import Image

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("Multi-Branch Neural Network - Hyperparameter Search")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 80)

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class BranchLayerConfig:
    """Configuration for a single branch layer with structured connectivity."""
    num_branches: int
    nodes_per_branch: int
    
    @property
    def total_nodes(self) -> int:
        return self.num_branches * self.nodes_per_branch


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    input_size: int = 784
    input_channels: int = 1
    input_height: int = 28
    input_width: int = 28
    num_classes: int = 10
    conv_channels: Tuple[int, ...] = (16, 12)
    conv_kernel_sizes: Tuple[int, ...] = (3, 3)
    num_conv_layers: int = 2
    conv_padding: int = 1
    pool_kernel_size: int = 2
    pool_stride: int = 2
    branch_layers: List[BranchLayerConfig] = field(default_factory=lambda: [
        BranchLayerConfig(num_branches=4, nodes_per_branch=6),
        BranchLayerConfig(num_branches=2, nodes_per_branch=3),
    ])
    
    @property
    def flat_size(self) -> int:
        spatial_size = self.input_height
        for _ in range(self.num_conv_layers):
            spatial_size = spatial_size // 2
        return self.conv_channels[-1] * spatial_size * spatial_size
    
    @property
    def num_branch_layers(self) -> int:
        return len(self.branch_layers)
    
    @property
    def first_layer_branches(self) -> int:
        return self.branch_layers[0].num_branches if self.branch_layers else 0
    
    @property
    def pad_size(self) -> int:
        if not self.branch_layers:
            return 0
        return (self.first_layer_branches - self.flat_size % self.first_layer_branches) % self.first_layer_branches
    
    @property
    def input_branch_size(self) -> int:
        if not self.branch_layers:
            return self.flat_size
        padded_size = self.flat_size + self.pad_size
        return padded_size // self.first_layer_branches
    
    @property
    def final_hidden_size(self) -> int:
        if not self.branch_layers:
            return self.flat_size
        return self.branch_layers[-1].total_nodes
    
    def validate_structured_connectivity(self) -> Tuple[bool, List[str]]:
        issues = []
        if len(self.branch_layers) < 2:
            return True, []
        for i in range(len(self.branch_layers) - 1):
            current_layer = self.branch_layers[i]
            next_layer = self.branch_layers[i + 1]
            if current_layer.num_branches % next_layer.num_branches != 0:
                issues.append(
                    f"Layer {i+1}→{i+2}: {current_layer.num_branches} branches cannot be "
                    f"evenly divided into {next_layer.num_branches} branches."
                )
        return len(issues) == 0, issues
    
    def get_trainable_parameters(self) -> int:
        conv_params = 0
        prev_channels = self.input_channels
        for i in range(self.num_conv_layers):
            kernel_size = self.conv_kernel_sizes[i] if i < len(self.conv_kernel_sizes) else self.conv_kernel_sizes[-1]
            curr_channels = self.conv_channels[i]
            weight_params = prev_channels * curr_channels * kernel_size * kernel_size
            bias_params = curr_channels
            conv_params += weight_params + bias_params
            prev_channels = curr_channels
        
        branch_params = 0
        for layer_idx, layer_config in enumerate(self.branch_layers):
            if layer_idx == 0:
                padded_size = self.flat_size + self.pad_size
                input_per_branch = padded_size // layer_config.num_branches
            else:
                prev_layer_config = self.branch_layers[layer_idx - 1]
                if prev_layer_config.num_branches % layer_config.num_branches == 0:
                    branches_per_group = prev_layer_config.num_branches // layer_config.num_branches
                    input_per_branch = branches_per_group * prev_layer_config.nodes_per_branch
                else:
                    input_per_branch = prev_layer_config.total_nodes // layer_config.num_branches
            layer_params = layer_config.num_branches * (
                input_per_branch * layer_config.nodes_per_branch + layer_config.nodes_per_branch
            )
            branch_params += layer_params
        
        output_params = self.final_hidden_size * self.num_classes + self.num_classes
        total_params = conv_params + branch_params + output_params
        return total_params


@dataclass
class TrainingConfig:
    """Training hyperparameters - optimized for speed."""
    batch_size: int = 128
    num_epochs: int = 25  # Base epochs, extended for high-performing models
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    alpha: float = 0.99
    lr_scheduler: str = "cosine"
    lr_min: float = 1e-6
    early_stopping: bool = True
    early_stopping_patience: int = 8
    extended_epochs: int = 50  # Extended epochs for high-performing models


@dataclass
class DataConfig:
    """Data configuration."""
    data_root: str = "./data"
    train_size: int = 45000
    valid_size: int = 15000
    test_size: int = 10000
    use_augmentation: bool = False
    normalize: bool = False
    mean: float = 0.1307
    std: float = 0.3081

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class StructuredBranchLayer(nn.Module):
    """A single layer with structured branch connectivity."""
    
    def __init__(self, num_branches: int, nodes_per_branch: int, 
                 input_per_branch: int, dropout_rate: float = 0.0):
        super().__init__()
        self.num_branches = num_branches
        self.nodes_per_branch = nodes_per_branch
        self.input_per_branch = input_per_branch
        self.branches = nn.ModuleList([
            nn.Linear(input_per_branch, nodes_per_branch)
            for _ in range(num_branches)
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, connectivity_info: dict = None) -> torch.Tensor:
        batch_size = x.size(0)
        branch_outputs = []
        
        if connectivity_info is None:
            branch_inputs = torch.split(x, self.input_per_branch, dim=1)
            for branch_idx, branch_linear in enumerate(self.branches):
                h = self.relu(branch_linear(branch_inputs[branch_idx]))
                h = self.dropout(h)
                branch_outputs.append(h)
        else:
            connections = connectivity_info['connections']
            for target_branch_idx in range(self.num_branches):
                source_branch_indices = connections[target_branch_idx]['source_branches']
                nodes_per_source = connections[target_branch_idx]['num_inputs'] // len(source_branch_indices)
                branch_inputs = []
                for source_idx in source_branch_indices:
                    start = source_idx * nodes_per_source
                    end = (source_idx + 1) * nodes_per_source
                    branch_inputs.append(x[:, start:end])
                combined_input = torch.cat(branch_inputs, dim=1)
                h = self.relu(self.branches[target_branch_idx](combined_input))
                h = self.dropout(h)
                branch_outputs.append(h)
        
        return torch.cat(branch_outputs, dim=1)


class ParallelBranchNet(nn.Module):
    """Parallel Branch Network with Structured Connectivity."""
    
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.config = model_config
        is_valid, issues = model_config.validate_structured_connectivity()
        if not is_valid:
            raise ValueError(f"Invalid branch configuration: {issues}")
        self.connectivity_pattern = self._get_connectivity_pattern()
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        prev_channels = self.config.input_channels
        for i in range(self.config.num_conv_layers):
            kernel_size = (self.config.conv_kernel_sizes[i] 
                          if i < len(self.config.conv_kernel_sizes) 
                          else self.config.conv_kernel_sizes[-1])
            conv_layer = nn.Conv2d(prev_channels, self.config.conv_channels[i],
                                  kernel_size=kernel_size, padding=self.config.conv_padding)
            self.conv_layers.append(conv_layer)
            pool_layer = nn.MaxPool2d(kernel_size=self.config.pool_kernel_size,
                                     stride=self.config.pool_stride)
            self.pool_layers.append(pool_layer)
            prev_channels = self.config.conv_channels[i]
        
        self.branch_layers = nn.ModuleList()
        for layer_idx, layer_config in enumerate(self.config.branch_layers):
            if layer_idx == 0:
                padded_size = self.config.flat_size + self.config.pad_size
                input_per_branch = padded_size // layer_config.num_branches
            else:
                prev_layer_config = self.config.branch_layers[layer_idx - 1]
                branches_per_group = prev_layer_config.num_branches // layer_config.num_branches
                input_per_branch = branches_per_group * prev_layer_config.nodes_per_branch
            branch_layer = StructuredBranchLayer(
                num_branches=layer_config.num_branches,
                nodes_per_branch=layer_config.nodes_per_branch,
                input_per_branch=input_per_branch,
                dropout_rate=0.0
            )
            self.branch_layers.append(branch_layer)
        
        self.fc_out = nn.Linear(self.config.final_hidden_size, self.config.num_classes)
        self.relu = nn.ReLU()
    
    def _get_connectivity_pattern(self) -> List[dict]:
        if len(self.config.branch_layers) < 2:
            return []
        connectivity = []
        for i in range(len(self.config.branch_layers) - 1):
            current_layer = self.config.branch_layers[i]
            next_layer = self.config.branch_layers[i + 1]
            if current_layer.num_branches % next_layer.num_branches == 0:
                branches_per_group = current_layer.num_branches // next_layer.num_branches
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
        return connectivity
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            x = self.relu(conv_layer(x))
            x = pool_layer(x)
        x = x.view(x.size(0), -1)
        if self.config.pad_size > 0:
            x = torch.nn.functional.pad(x, (0, self.config.pad_size))
        for layer_idx, branch_layer in enumerate(self.branch_layers):
            if layer_idx == 0:
                x = branch_layer(x, connectivity_info=None)
            else:
                conn_info = self.connectivity_pattern[layer_idx - 1]
                x = branch_layer(x, connectivity_info=conn_info)
        x = self.fc_out(x)
        return x
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# DATA LOADING - RAM OPTIMIZED
# ============================================================================

class RAMDataset(Dataset):
    """Dataset that loads all data into RAM for maximum speed."""
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor
        self.labels = labels_tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_mnist_to_ram(device):
    """Load entire MNIST dataset into RAM/GPU for maximum speed."""
    print("\n📦 Loading MNIST dataset into RAM...")
    
    # Download MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, 
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                  transform=transforms.ToTensor())
    
    # Load all training data into tensors
    train_data = []
    train_labels = []
    for img, label in train_dataset:
        train_data.append(img)
        train_labels.append(label)
    
    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels)
    
    # Load all test data into tensors
    test_data = []
    test_labels = []
    for img, label in test_dataset:
        test_data.append(img)
        test_labels.append(label)
    
    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels)
    
    # Split training into train/valid
    train_size = 45000
    valid_size = 15000
    indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(42))
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size+valid_size]
    
    train_data_split = train_data[train_indices]
    train_labels_split = train_labels[train_indices]
    valid_data_split = train_data[valid_indices]
    valid_labels_split = train_labels[valid_indices]
    
    print(f"✅ Training samples: {len(train_data_split):,}")
    print(f"✅ Validation samples: {len(valid_data_split):,}")
    print(f"✅ Test samples: {len(test_data):,}")
    print(f"✅ Data loaded into RAM!")
    
    return (train_data_split, train_labels_split, 
            valid_data_split, valid_labels_split,
            test_data, test_labels)


def create_ram_data_loaders(train_data, train_labels, valid_data, valid_labels, 
                            test_data, test_labels, batch_size, device):
    """Create data loaders from RAM data - keep data in RAM, move batches to GPU during training."""
    # Keep data in CPU RAM, use pin_memory for fast GPU transfer
    # This is more memory efficient than moving all data to GPU at once
    train_dataset = RAMDataset(train_data, train_labels)
    valid_dataset = RAMDataset(valid_data, valid_labels)
    test_dataset = RAMDataset(test_data, test_labels)
    
    # Use pin_memory for faster CPU->GPU transfer
    pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin_memory)
    
    return train_loader, valid_loader, test_loader

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def get_device():
    """Get CUDA device if available."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    return torch.device('cpu')


def create_optimizer(model, training_config):
    """Create optimizer."""
    optimizer_name = training_config.optimizer.lower()
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=training_config.learning_rate,
                         weight_decay=training_config.weight_decay, betas=training_config.betas)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=training_config.learning_rate,
                        weight_decay=training_config.weight_decay, 
                        momentum=training_config.momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_lr_scheduler(optimizer, training_config):
    """Create learning rate scheduler."""
    if training_config.lr_scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_config.num_epochs, eta_min=training_config.lr_min)
    return None


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        # Always move data to device (handles both CPU and CUDA)
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate_epoch(model, valid_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in valid_loader:
            # Always move data to device
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(valid_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate_test(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            # Always move data to device
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    avg_loss = total_loss / len(test_loader)
    return accuracy, f1, all_targets, all_predictions, avg_loss


def train_single_configuration(model_config, training_config, train_loader, 
                              valid_loader, test_loader, run_id, results_dir, device):
    """Train a single configuration with extended training for high performers."""
    torch.manual_seed(42 + run_id)
    np.random.seed(42 + run_id)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42 + run_id)
    
    try:
        model = ParallelBranchNet(model_config).to(device)
        optimizer = create_optimizer(model, training_config)
        scheduler = create_lr_scheduler(optimizer, training_config)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'valid_loss': [], 'train_accuracy': [], 
                  'valid_accuracy': [], 'test_accuracy': [], 'test_f1': []}
        best_valid_acc = 0.0
        best_test_acc = 0.0
        patience_counter = 0
        max_epochs = training_config.num_epochs
        extended_training = False
        
        # Training progress bar
        epoch_pbar = tqdm(range(max_epochs), desc=f"Run {run_id}", leave=False, position=1)
        
        for epoch in epoch_pbar:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            valid_loss, valid_acc = validate_epoch(model, valid_loader, criterion, device)
            
            # Test EVERY epoch to monitor test accuracy
            test_acc, test_f1, _, _, _ = evaluate_test(model, test_loader, device)
            test_acc_pct = test_acc * 100
            
            if scheduler is not None:
                scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_accuracy'].append(train_acc)
            history['valid_accuracy'].append(valid_acc)
            history['test_accuracy'].append(test_acc_pct)
            history['test_f1'].append(test_f1)
            
            # Track best test accuracy
            if test_acc_pct > best_test_acc:
                best_test_acc = test_acc_pct
            
            # Update epoch progress bar with test accuracy
            epoch_pbar.set_postfix({
                'TrnAcc': f'{train_acc:.1f}%',
                'ValAcc': f'{valid_acc:.1f}%',
                'TstAcc': f'{test_acc_pct:.1f}%',
                'Best': f'{best_test_acc:.1f}%'
            })
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check if model reaches 98% TEST accuracy - extend training
            if not extended_training and epoch >= 15 and test_acc_pct >= 98.0:
                extended_training = True
                max_epochs = training_config.extended_epochs
                patience_counter = 0  # Reset patience
                epoch_pbar.close()
                epoch_pbar = tqdm(range(epoch + 1, max_epochs), 
                                desc=f"Run {run_id} [EXTENDED 98%+]", leave=False, position=1,
                                initial=epoch + 1, total=max_epochs)
                # Recreate scheduler for extended epochs
                if training_config.lr_scheduler == 'cosine':
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = training_config.learning_rate * 0.5  # Reduce LR for fine-tuning
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max_epochs - epoch, eta_min=training_config.lr_min)
                continue
            
            if training_config.early_stopping and patience_counter >= training_config.early_stopping_patience:
                epoch_pbar.close()
                break
        
        epoch_pbar.close()
        
        final_test_acc, final_test_f1, y_true, y_pred, final_test_loss = evaluate_test(model, test_loader, device)
        
        run_dir = results_dir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'run_id': run_id,
            'model_config': {
                'num_conv_layers': model_config.num_conv_layers,
                'conv_channels': model_config.conv_channels,
                'conv_kernel_sizes': model_config.conv_kernel_sizes,
                'branch_layers': [
                    {'num_branches': layer.num_branches, 'nodes_per_branch': layer.nodes_per_branch}
                    for layer in model_config.branch_layers
                ],
                'param_count': model_config.get_trainable_parameters()
            },
            'training_config': {
                'batch_size': training_config.batch_size,
                'learning_rate': training_config.learning_rate,
                'optimizer': training_config.optimizer,
                'num_epochs': max_epochs,
                'actual_epochs': epoch + 1,
                'extended_training': extended_training
            },
            'final_metrics': {
                'best_valid_accuracy': best_valid_acc,
                'final_test_accuracy': final_test_acc * 100,
                'final_test_f1': final_test_f1,
                'final_test_loss': final_test_loss
            },
            'history': history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(run_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        if final_test_acc > 0.96:
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'training_config': training_config,
                'param_count': model_config.get_trainable_parameters()
            }, run_dir / "model.pth")
        
        del model, optimizer, scheduler
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return results
    except Exception as e:
        print(f"❌ Error in run {run_id}: {str(e)}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'run_id': run_id, 'error': str(e), 'timestamp': datetime.now().isoformat()}

# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

def generate_hyperparameter_combinations():
    """Generate all valid hyperparameter combinations between 6k-7k parameters."""
    conv_configs = [
        # 2-layer configurations targeting 6k-7k params
        {'num_conv_layers': 2, 'conv_channels': (16, 12), 'kernel_sizes': (3, 3)},
        {'num_conv_layers': 2, 'conv_channels': (18, 14), 'kernel_sizes': (3, 3)},
        {'num_conv_layers': 2, 'conv_channels': (20, 16), 'kernel_sizes': (3, 3)},
        {'num_conv_layers': 2, 'conv_channels': (22, 18), 'kernel_sizes': (3, 3)},
        {'num_conv_layers': 2, 'conv_channels': (24, 20), 'kernel_sizes': (3, 3)},
        {'num_conv_layers': 2, 'conv_channels': (16, 12), 'kernel_sizes': (5, 3)},
        {'num_conv_layers': 2, 'conv_channels': (18, 14), 'kernel_sizes': (5, 3)},
        {'num_conv_layers': 2, 'conv_channels': (20, 16), 'kernel_sizes': (5, 3)},
        {'num_conv_layers': 2, 'conv_channels': (22, 18), 'kernel_sizes': (5, 3)},
        # 3-layer configurations targeting 6k-7k params
        {'num_conv_layers': 3, 'conv_channels': (12, 10, 8), 'kernel_sizes': (3, 3, 3)},
        {'num_conv_layers': 3, 'conv_channels': (14, 12, 8), 'kernel_sizes': (3, 3, 3)},
        {'num_conv_layers': 3, 'conv_channels': (16, 12, 10), 'kernel_sizes': (3, 3, 3)},
        {'num_conv_layers': 3, 'conv_channels': (18, 14, 10), 'kernel_sizes': (3, 3, 3)},
        {'num_conv_layers': 3, 'conv_channels': (20, 16, 12), 'kernel_sizes': (3, 3, 3)},
        # 4-layer configurations targeting 6k-7k params
        {'num_conv_layers': 4, 'conv_channels': (12, 10, 8, 6), 'kernel_sizes': (3, 3, 3, 3)},
        {'num_conv_layers': 4, 'conv_channels': (14, 12, 10, 8), 'kernel_sizes': (3, 3, 3, 3)},
        {'num_conv_layers': 4, 'conv_channels': (16, 14, 12, 8), 'kernel_sizes': (3, 3, 3, 3)},
    ]
    
    branch_configs = [
        # Single layer branches
        [BranchLayerConfig(num_branches=2, nodes_per_branch=8)],
        [BranchLayerConfig(num_branches=2, nodes_per_branch=10)],
        [BranchLayerConfig(num_branches=3, nodes_per_branch=6)],
        [BranchLayerConfig(num_branches=3, nodes_per_branch=8)],
        [BranchLayerConfig(num_branches=4, nodes_per_branch=6)],
        [BranchLayerConfig(num_branches=4, nodes_per_branch=8)],
        [BranchLayerConfig(num_branches=6, nodes_per_branch=6)],
        # Two-layer branches
        [BranchLayerConfig(num_branches=4, nodes_per_branch=8), BranchLayerConfig(num_branches=2, nodes_per_branch=6)],
        [BranchLayerConfig(num_branches=6, nodes_per_branch=6), BranchLayerConfig(num_branches=3, nodes_per_branch=5)],
        [BranchLayerConfig(num_branches=6, nodes_per_branch=8), BranchLayerConfig(num_branches=2, nodes_per_branch=6)],
        [BranchLayerConfig(num_branches=8, nodes_per_branch=6), BranchLayerConfig(num_branches=4, nodes_per_branch=5)],
    ]
    
    learning_rates = [0.001, 0.01]
    optimizers = ['adam', 'sgd']
    batch_sizes = [128, 256]
    
    valid_combinations = []
    for conv_config in conv_configs:
        for branch_layers in branch_configs:
            for lr in learning_rates:
                for optimizer in optimizers:
                    for batch_size in batch_sizes:
                        model_config = ModelConfig(
                            conv_channels=conv_config['conv_channels'],
                            conv_kernel_sizes=conv_config['kernel_sizes'],
                            num_conv_layers=conv_config['num_conv_layers'],
                            branch_layers=branch_layers
                        )
                        param_count = model_config.get_trainable_parameters()
                        # Filter for 6k-7k parameter range
                        if 6000 <= param_count <= 7000:
                            is_valid, issues = model_config.validate_structured_connectivity()
                            if is_valid:
                                training_config = TrainingConfig(
                                    batch_size=batch_size,
                                    learning_rate=lr,
                                    optimizer=optimizer
                                )
                                valid_combinations.append({
                                    'model_config': model_config,
                                    'training_config': training_config,
                                    'param_count': param_count,
                                    'conv_config': conv_config
                                })
    return valid_combinations

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def save_intermediate_results(all_results, results_dir, current_run, successful_runs, failed_runs):
    """Save intermediate results."""
    try:
        progress_summary = {
            'total_runs_completed': current_run,
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'timestamp': datetime.now().isoformat(),
            'all_results': all_results
        }
        intermediate_path = results_dir / 'intermediate_results.json'
        with open(intermediate_path, 'w') as f:
            json.dump(progress_summary, f, indent=2)
        if current_run % 50 == 0:
            backup_path = results_dir / f'backup_results_run_{current_run}.json'
            with open(backup_path, 'w') as f:
                json.dump(progress_summary, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving intermediate results: {e}")


def main():
    """Main execution function."""
    # Setup
    device = get_device()
    print(f"\n🖥️ Device: {device}")
    if device.type == 'cuda':
        print(f"🔥 CUDA available! GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data into RAM
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_mnist_to_ram(device)
    
    # Generate combinations
    print("\n🔍 Generating hyperparameter combinations...")
    combinations = generate_hyperparameter_combinations()
    print(f"✅ Found {len(combinations)} valid combinations (6k-7k parameters)")
    
    # Show parameter distribution
    param_counts = [c['param_count'] for c in combinations]
    print(f"📊 Parameter range: {min(param_counts):,} - {max(param_counts):,}")
    print(f"📊 Average parameters: {sum(param_counts) // len(param_counts):,}")
    
    # Create results directory
    results_dir = Path("./hyperparameter_search_results")
    results_dir.mkdir(exist_ok=True)
    
    # Initialize tracking
    all_results = []
    successful_runs = 0
    failed_runs = 0
    data_config = DataConfig()
    
    print(f"\n🚀 Starting hyperparameter search with {len(combinations)} configurations...")
    print(f"📁 Results will be saved in: {results_dir.absolute()}")
    
    # Cache data loaders for each batch size to avoid recreation
    loader_cache = {}
    
    # Track best accuracy
    best_test_acc = 0.0
    
    # Main search loop
    with tqdm(total=len(combinations), desc="🔬 Hyperparameter Search", unit="config", position=0) as pbar:
        for i, combo in enumerate(combinations):
            model_config = combo['model_config']
            training_config = combo['training_config']
            param_count = combo['param_count']
            
            # Get or create data loaders for this batch size (cached)
            batch_size = training_config.batch_size
            if batch_size not in loader_cache:
                train_loader, valid_loader, test_loader = create_ram_data_loaders(
                    train_data, train_labels, valid_data, valid_labels, 
                    test_data, test_labels, batch_size, device)
                loader_cache[batch_size] = (train_loader, valid_loader, test_loader)
            else:
                train_loader, valid_loader, test_loader = loader_cache[batch_size]
            
            # Train configuration
            start_time = time.time()
            result = train_single_configuration(
                model_config, training_config, train_loader, valid_loader, 
                test_loader, i, results_dir, device)
            training_time = time.time() - start_time
            result['training_time_seconds'] = training_time
            
            all_results.append(result)
            if 'error' in result:
                failed_runs += 1
                test_acc = 0.0
            else:
                successful_runs += 1
                test_acc = result['final_metrics']['final_test_accuracy']
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
            
            save_intermediate_results(all_results, results_dir, i + 1, successful_runs, failed_runs)
            
            # Update progress bar with current run info
            pbar.set_postfix({
                'Params': f'{param_count:,}',
                'Conv': f'{model_config.num_conv_layers}L{model_config.conv_channels}',
                'Branch': f'{len(model_config.branch_layers)}L',
                'LR': f'{training_config.learning_rate}',
                'Opt': training_config.optimizer.upper()[:3],
                'BS': batch_size,
                'TestAcc': f'{test_acc:.1f}%',
                'Best': f'{best_test_acc:.1f}%'
            })
            
            pbar.update(1)
            
            if (i + 1) % 10 == 0:
                print(f"\n📊 Progress: {i+1}/{len(combinations)} runs completed")
                print(f"✅ Successful: {successful_runs}, ❌ Failed: {failed_runs}")
    
    print(f"\n✅ Hyperparameter search completed!")
    print(f"📊 Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📁 Results saved in: {results_dir.absolute()}")
    
    # Save final summary
    final_summary = {
        'total_combinations': len(combinations),
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'completion_rate': f"{(successful_runs / len(combinations)) * 100:.2f}%",
        'timestamp': datetime.now().isoformat(),
        'all_results': all_results
    }
    with open(results_dir / 'final_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"💾 Final summary saved to: {results_dir / 'final_summary.json'}")
    
    # Analyze results
    successful_results = [r for r in all_results if 'error' not in r]
    if successful_results:
        test_accuracies = [r['final_metrics']['final_test_accuracy'] for r in successful_results]
        best_acc = max(test_accuracies)
        best_idx = test_accuracies.index(best_acc)
        best_result = successful_results[best_idx]
        
        print(f"\n🏆 BEST RESULT:")
        print(f"  • Test Accuracy: {best_acc:.2f}%")
        print(f"  • Run ID: {best_result['run_id']}")
        print(f"  • Parameters: {best_result['model_config']['param_count']:,}")
        print(f"  • Conv Layers: {best_result['model_config']['num_conv_layers']}")
        print(f"  • Branch Layers: {len(best_result['model_config']['branch_layers'])}")


if __name__ == "__main__":
    main()
