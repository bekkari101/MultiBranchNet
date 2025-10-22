"""
Plotting utilities for training visualization.

Creates training curves, confusion matrix, and other visualizations
for the Parallel Branch Network experiments.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI windows
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from sklearn.metrics import confusion_matrix, classification_report
import torch


def plot_training_curves(
    history: Dict[str, List[float]],
    run_dir: Path,
    run_number: int,
    save_plot: bool = True
) -> Path:
    """Plot training and validation loss/accuracy curves."""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Progress - Run {run_number}', fontsize=16, fontweight='bold')
    
    epochs = history.get('epoch', list(range(1, len(history.get('train_loss', [])) + 1)))
    
    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['valid_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    if 'test_accuracy' in history:
        axes[0, 1].plot(epochs, history['test_accuracy'], 'g-', label='Test Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score curve
    if 'test_f1' in history:
        axes[1, 0].plot(epochs, history['test_f1'], 'g-', label='Test F1 Score', linewidth=2)
        axes[1, 0].set_title('F1 Score Curve', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning rate curve (if available)
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], 'purple', label='Learning Rate', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    else:
        # Show final metrics as text
        final_metrics = {
            'Final Train Loss': f"{history['train_loss'][-1]:.4f}",
            'Final Valid Loss': f"{history['valid_loss'][-1]:.4f}",
            'Final Train Acc': f"{history['train_accuracy'][-1]:.4f}",
            'Final Valid Acc': f"{history['valid_accuracy'][-1]:.4f}",
        }
        if 'test_accuracy' in history:
            final_metrics['Final Test Acc'] = f"{history['test_accuracy'][-1]:.4f}"
        if 'test_f1' in history:
            final_metrics['Final Test F1'] = f"{history['test_f1'][-1]:.4f}"
        
        axes[1, 1].text(0.1, 0.5, '\n'.join([f"{k}: {v}" for k, v in final_metrics.items()]),
                       transform=axes[1, 1].transAxes, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 1].set_title('Final Metrics', fontweight='bold')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = run_dir / "plots" / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to: {plot_path}")
    
    plt.close()  # Close plot to free memory and prevent display
    
    return plot_path if save_plot else None


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    run_dir: Path,
    run_number: int,
    class_names: Optional[List[str]] = None,
    save_plot: bool = True
) -> Path:
    """Plot confusion matrix with detailed metrics."""
    
    if class_names is None:
        class_names = [str(i) for i in range(10)]  # MNIST digits 0-9
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Confusion Matrix - Run {run_number}', fontsize=16, fontweight='bold')
    
    # Plot 1: Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'shrink': 0.8})
    axes[0].set_title('Confusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Plot 2: Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'shrink': 0.8})
    axes[1].set_title('Normalized Confusion Matrix', fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = run_dir / "plots" / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {plot_path}")
    
    plt.close()  # Close plot to free memory and prevent display
    
    return plot_path if save_plot else None


def plot_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    run_dir: Path,
    run_number: int,
    class_names: Optional[List[str]] = None,
    save_plot: bool = True
) -> Path:
    """Plot per-class precision, recall, and F1 scores."""
    
    if class_names is None:
        class_names = [str(i) for i in range(10)]  # MNIST digits 0-9
    
    # Calculate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract metrics for each class
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for class_name in class_names:
        if class_name in report:
            classes.append(class_name)
            precision.append(report[class_name]['precision'])
            recall.append(report[class_name]['recall'])
            f1.append(report[class_name]['f1-score'])
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Metrics - Run {run_number}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = run_dir / "plots" / "class_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Class metrics saved to: {plot_path}")
    
    plt.close()  # Close plot to free memory and prevent display
    
    return plot_path if save_plot else None


def create_all_plots(
    history: Dict[str, List[float]],
    y_true: List[int],
    y_pred: List[int],
    run_dir: Path,
    run_number: int,
    class_names: Optional[List[str]] = None
) -> Dict[str, Path]:
    """Create all plots and return their paths."""
    
    plots = {}
    
    # Training curves
    plots['training_curves'] = plot_training_curves(history, run_dir, run_number)
    
    # Confusion matrix
    plots['confusion_matrix'] = plot_confusion_matrix(y_true, y_pred, run_dir, run_number, class_names)
    
    # Class metrics
    plots['class_metrics'] = plot_class_metrics(y_true, y_pred, run_dir, run_number, class_names)
    
    return plots


def load_and_plot_from_history(
    history_path: Path,
    y_true: List[int],
    y_pred: List[int],
    run_dir: Optional[Path] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Path]:
    """Load history from JSON and create all plots."""
    
    # Load history
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    history = history_data['history']
    run_number = history_data['run_number']
    
    if run_dir is None:
        run_dir = history_path.parent
    
    return create_all_plots(history, y_true, y_pred, run_dir, run_number, class_names)


if __name__ == "__main__":
    # Test plotting with dummy data
    dummy_history = {
        'epoch': list(range(1, 21)),
        'train_loss': [0.5 - 0.02*i + 0.01*np.random.random() for i in range(20)],
        'valid_loss': [0.6 - 0.015*i + 0.02*np.random.random() for i in range(20)],
        'train_accuracy': [0.8 + 0.01*i + 0.005*np.random.random() for i in range(20)],
        'valid_accuracy': [0.75 + 0.008*i + 0.01*np.random.random() for i in range(20)],
        'test_f1': [0.7 + 0.01*i + 0.005*np.random.random() for i in range(20)]
    }
    
    dummy_y_true = np.random.randint(0, 10, 1000)
    dummy_y_pred = dummy_y_true + np.random.randint(-1, 2, 1000)
    dummy_y_pred = np.clip(dummy_y_pred, 0, 9)
    
    run_dir = Path("result/run_000")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    
    plots = create_all_plots(dummy_history, dummy_y_true.tolist(), dummy_y_pred.tolist(), 
                           run_dir, 0)
    print("Test plots created successfully!")
