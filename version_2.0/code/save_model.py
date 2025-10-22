"""
Model saving and loading utilities for Parallel Branch Network.

Handles saving/loading of model weights, optimizer states, training history,
and experiment metadata in JSON format.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn

from hyperP import ModelConfig, TrainingConfig, ExperimentConfig


def get_next_run_number(results_dir: Path) -> int:
    """Get the next run number by checking existing directories."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    existing_runs = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith("run_"):
            try:
                run_num = int(item.name.split("_")[1])
                existing_runs.append(run_num)
            except (ValueError, IndexError):
                continue
    
    if not existing_runs:
        return 0
    
    return max(existing_runs) + 1


def create_run_directory(results_dir: Path, run_number: int) -> Path:
    """Create a new run directory and return its path."""
    run_dir = results_dir / f"run_{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    
    return run_dir


def save_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    valid_accuracy: float,
    test_f1: float,
    run_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    experiment_config: ExperimentConfig,
    is_best: bool = False
) -> Dict[str, Any]:
    """Save model checkpoint with metadata."""
    
    # Prepare checkpoint data
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
        "test_f1": test_f1,
        "model_config": {
            "branches": model_config.branches,
            "branch_size": model_config.branch_size,
            "branch_layers": model_config.branch_layers,
            "conv_channels": model_config.conv_channels,
            "input_branch_size": model_config.input_branch_size,
            "hidden_size": model_config.hidden_size,
            "trainable_parameters": model_config.get_trainable_parameters()
        },
        "training_config": {
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "num_epochs": training_config.num_epochs,
            "optimizer": training_config.optimizer,
            "weight_decay": training_config.weight_decay
        },
        "experiment_config": {
            "experiment_name": experiment_config.experiment_name,
            "run_name": experiment_config.run_name,
            "seed": experiment_config.seed,
            "device": experiment_config.device
        },
        "timestamp": datetime.now().isoformat(),
        "is_best": is_best
    }
    
    # Save model checkpoint
    checkpoint_path = run_dir / "models" / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model separately
    if is_best:
        best_path = run_dir / "models" / "best_model.pth"
        torch.save(checkpoint, best_path)
    
    # Save latest model
    latest_path = run_dir / "models" / "latest_model.pth"
    torch.save(checkpoint, latest_path)
    
    return checkpoint


def load_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path
) -> Tuple[int, float, float, float, float]:
    """Load model from checkpoint and return training state."""
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Return training state
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]
    valid_accuracy = checkpoint["valid_accuracy"]
    test_f1 = checkpoint["test_f1"]
    
    return epoch, train_loss, valid_loss, valid_accuracy, test_f1


def save_training_history(
    history: Dict[str, list],
    run_dir: Path,
    run_number: int
) -> Path:
    """Save training history to JSON file."""
    
    # Add metadata
    history_data = {
        "run_number": run_number,
        "timestamp": datetime.now().isoformat(),
        "total_epochs": len(history.get("epoch", [])),
        "history": history
    }
    
    # Save to JSON
    history_path = run_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    return history_path


def load_training_history(history_path: Path) -> Dict[str, Any]:
    """Load training history from JSON file."""
    
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    return history_data


def save_experiment_summary(
    run_dir: Path,
    run_number: int,
    final_metrics: Dict[str, float],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    experiment_config: ExperimentConfig,
    training_time: float
) -> Path:
    """Save experiment summary with all key information."""
    
    summary = {
        "run_number": run_number,
        "timestamp": datetime.now().isoformat(),
        "training_time_seconds": training_time,
        "final_metrics": final_metrics,
        "model_config": {
            "branches": model_config.branches,
            "branch_size": model_config.branch_size,
            "branch_layers": model_config.branch_layers,
            "conv_channels": model_config.conv_channels,
            "input_branch_size": model_config.input_branch_size,
            "hidden_size": model_config.hidden_size,
            "trainable_parameters": model_config.get_trainable_parameters()
        },
        "training_config": {
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "num_epochs": training_config.num_epochs,
            "optimizer": training_config.optimizer,
            "weight_decay": training_config.weight_decay,
            "early_stopping": training_config.early_stopping,
            "early_stopping_patience": training_config.early_stopping_patience
        },
        "experiment_config": {
            "experiment_name": experiment_config.experiment_name,
            "run_name": experiment_config.run_name,
            "seed": experiment_config.seed,
            "device": experiment_config.device
        }
    }
    
    # Save summary
    summary_path = run_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_path


def print_save_info(run_dir: Path, run_number: int, final_metrics: Dict[str, float]):
    """Print information about saved files."""
    
    print("=" * 60)
    print("SAVE INFORMATION")
    print("=" * 60)
    
    print(f"\n📁 Run Directory: {run_dir}")
    print(f"🔢 Run Number: {run_number}")
    
    print(f"\n💾 Saved Files:")
    print(f"  • Models: {run_dir / 'models'}")
    print(f"    - best_model.pth")
    print(f"    - latest_model.pth")
    print(f"    - checkpoint_epoch_*.pth")
    
    print(f"  • Data: {run_dir}")
    print(f"    - training_history.json")
    print(f"    - experiment_summary.json")
    
    print(f"  • Plots: {run_dir / 'plots'}")
    print(f"    - training_curves.png")
    print(f"    - confusion_matrix.png")
    
    print(f"\n📊 Final Metrics:")
    for metric, value in final_metrics.items():
        print(f"  • {metric}: {value:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test run numbering
    results_dir = Path("result")
    next_run = get_next_run_number(results_dir)
    print(f"Next run number: {next_run}")
    
    # Test directory creation
    run_dir = create_run_directory(results_dir, next_run)
    print(f"Created run directory: {run_dir}")
