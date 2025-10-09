"""
Data loading module for MNIST Parallel Branch Network.

Handles loading of train/valid/test datasets with proper transforms,
data augmentation (optional), and batch processing.
"""

import os
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from hyperP import DataConfig, ModelConfig


class MNISTImageDataset(Dataset):
    """Custom dataset for loading MNIST images from structured folders with RAM caching."""
    
    def __init__(self, data_dir: Path, transform: Optional[transforms.Compose] = None, load_to_ram: bool = True, progress_callback=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.load_to_ram = load_to_ram
        self.samples = []
        self.labels = []
        self.images_in_ram = []  # Store images in RAM
        
        print(f"Loading dataset from {data_dir}...")
        if load_to_ram:
            print("🔄 Loading all images into RAM for faster training...")
        
        # Count total images first
        total_images = 0
        for label_dir in sorted(self.data_dir.iterdir()):
            if label_dir.is_dir():
                total_images += len(list(label_dir.glob("*.png")))
        
        # Load all images and labels with progress
        loaded_count = 0
        update_interval = max(1, total_images // 100)  # Update every 1% or every image if < 100 images
        
        for label_dir in sorted(self.data_dir.iterdir()):
            if label_dir.is_dir():
                label = int(label_dir.name)
                for img_path in sorted(label_dir.glob("*.png")):
                    self.samples.append(str(img_path))
                    self.labels.append(label)
                    
                    if load_to_ram:
                        # Load image into RAM immediately
                        image = Image.open(img_path).convert('L')  # Grayscale
                        if self.transform:
                            image = self.transform(image)
                        self.images_in_ram.append(image)
                    
                    loaded_count += 1
                    # Throttle progress updates to prevent GUI freezing
                    if progress_callback and (loaded_count % update_interval == 0 or loaded_count == total_images):
                        progress_callback(loaded_count, total_images)
        
        if load_to_ram:
            print(f"✅ Loaded {len(self.images_in_ram)} images into RAM")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.labels[idx]
        
        if self.load_to_ram:
            # Return image from RAM
            image = self.images_in_ram[idx]
        else:
            # Load image from disk (original behavior)
            img_path = self.samples[idx]
            image = Image.open(img_path).convert('L')  # Grayscale
            if self.transform:
                image = self.transform(image)
        
        return image, label


def get_transforms(data_config: DataConfig, is_training: bool = True) -> transforms.Compose:
    """Create transforms for training or validation."""
    transform_list = []
    
    if is_training and data_config.use_augmentation:
        # Data augmentation for training
        transform_list.extend([
            transforms.RandomRotation(degrees=data_config.rotation_degrees),
            transforms.RandomAffine(
                degrees=0,
                translate=data_config.translate,
                scale=data_config.scale
            ),
            transforms.ColorJitter(
                brightness=data_config.brightness,
                contrast=data_config.contrast
            )
        ])
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalization
    if data_config.normalize:
        transform_list.append(
            transforms.Normalize(mean=[data_config.mean], std=[data_config.std])
        )
    
    return transforms.Compose(transform_list)


def create_data_loaders(
    data_config: DataConfig,
    model_config: ModelConfig,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    load_to_ram: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders with optional RAM loading."""
    
    # Use default values if not provided
    batch_size = batch_size or 128  # Default batch size
    num_workers = num_workers or 4  # Default num_workers
    
    # Get transforms
    train_transform = get_transforms(data_config, is_training=True)
    val_transform = get_transforms(data_config, is_training=False)
    
    # Create datasets
    data_root = Path(data_config.data_root)
    
    print("=" * 60)
    print("LOADING DATASETS INTO RAM")
    print("=" * 60)
    
    # Create progress bars for each dataset
    from tqdm import tqdm
    
    # Train dataset
    train_pbar = tqdm(total=1, desc="📚 Loading train dataset", unit="dataset", leave=False)
    def train_progress(loaded, total):
        train_pbar.set_postfix({'Images': f'{loaded}/{total}'})
        if loaded == total:
            train_pbar.update(1)
            train_pbar.close()
    
    train_dataset = MNISTImageDataset(
        data_root / data_config.train_dir,
        transform=train_transform,
        load_to_ram=load_to_ram,
        progress_callback=train_progress
    )
    
    # Valid dataset
    valid_pbar = tqdm(total=1, desc="📖 Loading valid dataset", unit="dataset", leave=False)
    def valid_progress(loaded, total):
        valid_pbar.set_postfix({'Images': f'{loaded}/{total}'})
        if loaded == total:
            valid_pbar.update(1)
            valid_pbar.close()
    
    valid_dataset = MNISTImageDataset(
        data_root / data_config.valid_dir,
        transform=val_transform,
        load_to_ram=load_to_ram,
        progress_callback=valid_progress
    )
    
    # Test dataset
    test_pbar = tqdm(total=1, desc="📝 Loading test dataset", unit="dataset", leave=False)
    def test_progress(loaded, total):
        test_pbar.set_postfix({'Images': f'{loaded}/{total}'})
        if loaded == total:
            test_pbar.update(1)
            test_pbar.close()
    
    test_dataset = MNISTImageDataset(
        data_root / data_config.test_dir,
        transform=val_transform,
        load_to_ram=load_to_ram,
        progress_callback=test_progress
    )
    
    # Adjust num_workers for RAM loading
    if load_to_ram:
        num_workers = 0  # No need for multiprocessing when data is in RAM
        print("📝 Using num_workers=0 since data is loaded into RAM")
    
    # Create data loaders (disable pin_memory for CPU training)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disabled for CPU training
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Disabled for CPU training
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Disabled for CPU training
        drop_last=False
    )
    
    print("✅ Data loaders created successfully!")
    print("=" * 60)
    
    return train_loader, valid_loader, test_loader


def get_dataset_info(data_config: DataConfig) -> Dict[str, Any]:
    """Get information about the datasets."""
    data_root = Path(data_config.data_root)
    
    info = {
        "data_root": str(data_root),
        "splits": {
            "train": {"dir": data_config.train_dir, "expected_size": data_config.train_size},
            "valid": {"dir": data_config.valid_dir, "expected_size": data_config.valid_size},
            "test": {"dir": data_config.test_dir, "expected_size": data_config.test_size}
        },
        "augmentation": {
            "enabled": data_config.use_augmentation,
            "rotation_degrees": data_config.rotation_degrees,
            "translate": data_config.translate,
            "scale": data_config.scale,
            "brightness": data_config.brightness,
            "contrast": data_config.contrast
        },
        "normalization": {
            "enabled": data_config.normalize,
            "mean": data_config.mean,
            "std": data_config.std
        }
    }
    
    # Check actual dataset sizes
    for split_name, split_info in info["splits"].items():
        split_dir = data_root / split_info["dir"]
        if split_dir.exists():
            actual_size = len(list(split_dir.rglob("*.png")))
            split_info["actual_size"] = actual_size
            split_info["size_match"] = actual_size == split_info["expected_size"]
        else:
            split_info["actual_size"] = 0
            split_info["size_match"] = False
    
    return info


def print_data_info(data_config: DataConfig):
    """Print information about the datasets."""
    info = get_dataset_info(data_config)
    
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    print(f"\n📁 Data Root: {info['data_root']}")
    
    print(f"\n📊 Dataset Splits:")
    for split_name, split_info in info["splits"].items():
        status = "✅" if split_info["size_match"] else "❌"
        print(f"  {status} {split_name.upper()}: {split_info['actual_size']:,} / {split_info['expected_size']:,} images")
    
    print(f"\n🔄 Augmentation: {'Enabled' if info['augmentation']['enabled'] else 'Disabled'}")
    if info['augmentation']['enabled']:
        print(f"  • Rotation: ±{info['augmentation']['rotation_degrees']}°")
        print(f"  • Translation: {info['augmentation']['translate']}")
        print(f"  • Scale: {info['augmentation']['scale']}")
        print(f"  • Brightness: ±{info['augmentation']['brightness']}")
        print(f"  • Contrast: ±{info['augmentation']['contrast']}")
    
    print(f"\n📏 Normalization: {'Enabled' if info['normalization']['enabled'] else 'Disabled'}")
    if info['normalization']['enabled']:
        print(f"  • Mean: {info['normalization']['mean']}")
        print(f"  • Std: {info['normalization']['std']}")
    
    print("=" * 60)


if __name__ == "__main__":
    from hyperP import DEFAULT_DATA_CONFIG
    print_data_info(DEFAULT_DATA_CONFIG)
