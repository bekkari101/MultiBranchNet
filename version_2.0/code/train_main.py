"""
Main training script for Parallel Branch Network on MNIST with STRUCTURED CONNECTIVITY.

NEW FEATURES:
- Support for multiple optimizers (Adam, SGD, AdamW, RMSprop)
- Structured connectivity between branch layers
- Real-time progress tracking and plotting
- Enhanced experiment management

This script orchestrates the entire training pipeline including:
- Data loading and preprocessing
- Model training with progress tracking
- Model saving and loading
- Plotting and visualization
- Experiment management with run numbering
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# Limit PyTorch to use only 3 CPU cores
os.environ['OMP_NUM_THREADS'] = '3'
os.environ['MKL_NUM_THREADS'] = '3'
torch.set_num_threads(3)

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from hyperP import (
    DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_DATA_CONFIG, 
    DEFAULT_LOGGING_CONFIG, DEFAULT_EXPERIMENT_CONFIG, get_device
)
from model import ParallelBranchNet
from loadata import create_data_loaders, print_data_info
from save_model import (
    get_next_run_number, create_run_directory, save_model, save_training_history,
    save_experiment_summary, print_save_info
)
from plot import create_all_plots, plot_training_curves


def create_optimizer(model, training_config):
    """
    Create optimizer based on configuration.
    
    Supports multiple optimizer types:
    - adam: Adam optimizer
    - sgd: Stochastic Gradient Descent with momentum
    - adamw: AdamW optimizer (Adam with weight decay fix)
    - rmsprop: RMSprop optimizer
    """
    optimizer_name = training_config.optimizer.lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=training_config.betas
        )
        print(f"✅ Using Adam optimizer (lr={training_config.learning_rate}, betas={training_config.betas})")
    
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            momentum=training_config.momentum,
            nesterov=True
        )
        print(f"✅ Using SGD optimizer (lr={training_config.learning_rate}, momentum={training_config.momentum})")
    
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=training_config.betas
        )
        print(f"✅ Using AdamW optimizer (lr={training_config.learning_rate}, betas={training_config.betas})")
    
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            alpha=training_config.alpha,
            momentum=training_config.momentum
        )
        print(f"✅ Using RMSprop optimizer (lr={training_config.learning_rate}, alpha={training_config.alpha})")
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from: adam, sgd, adamw, rmsprop")
    
    return optimizer


def create_lr_scheduler(optimizer, training_config):
    """
    Create learning rate scheduler based on configuration.
    
    Supports:
    - cosine: Cosine annealing
    - step: Step decay
    - plateau: Reduce on plateau
    - none: No scheduling
    """
    scheduler_name = training_config.lr_scheduler.lower()
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.num_epochs,
            eta_min=training_config.lr_min
        )
        print(f"✅ Using Cosine Annealing LR scheduler")
    
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.lr_step_size,
            gamma=training_config.lr_gamma
        )
        print(f"✅ Using Step LR scheduler (step_size={training_config.lr_step_size}, gamma={training_config.lr_gamma})")
    
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=training_config.lr_patience,
            factor=training_config.lr_gamma,
            min_lr=training_config.lr_min
        )
        print(f"✅ Using ReduceLROnPlateau scheduler (patience={training_config.lr_patience})")
    
    elif scheduler_name == 'none':
        scheduler = None
        print(f"✅ No LR scheduler")
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}. Choose from: cosine, step, plateau, none")
    
    return scheduler


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
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
        pbar = tqdm(valid_loader, desc="Validation", leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(valid_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate_test(model, test_loader, device):
    """Evaluate on test set and return predictions and test loss."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    avg_loss = total_loss / max(1, len(test_loader))
    return accuracy, f1, all_targets, all_predictions, avg_loss


def train_model(
    model_config,
    training_config,
    data_config,
    experiment_config,
    run_dir: Path,
    run_number: int
):
    """Main training function with structured connectivity support."""
    
    # Set device
    device = get_device(experiment_config.device)
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(experiment_config.seed)
    np.random.seed(experiment_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(experiment_config.seed)
    
    # Create data loaders
    print("Loading data...")
    # Change to project root directory for data loading
    original_cwd = os.getcwd()
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    train_loader, valid_loader, test_loader = create_data_loaders(
        data_config, model_config,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers
    )
    
    # Change back to original directory
    os.chdir(original_cwd)
    
    # Print data info
    print_data_info(data_config)
    
    # Create model
    print("\n" + "=" * 80)
    print("CREATING MODEL WITH STRUCTURED CONNECTIVITY")
    print("=" * 80)
    model = ParallelBranchNet(model_config).to(device)
    model.print_architecture()
    
    estimated_params = model_config.get_trainable_parameters()
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model parameters -> estimated: {estimated_params:,}, actual: {actual_params:,}")
    
    # Create optimizer (MULTIPLE OPTIONS SUPPORTED!)
    print("\n" + "=" * 80)
    print("OPTIMIZER CONFIGURATION")
    print("=" * 80)
    optimizer = create_optimizer(model, training_config)
    
    # Create learning rate scheduler
    scheduler = create_lr_scheduler(optimizer, training_config)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'valid_loss': [],
        'train_accuracy': [],
        'valid_accuracy': [],
        'test_accuracy': [],
        'test_f1': [],
        'learning_rate': []
    }
    
    # Training loop
    best_valid_acc = 0.0
    patience_counter = 0
    start_time = time.time()
    
    print(f"\n" + "=" * 80)
    print(f"STARTING TRAINING FOR {training_config.num_epochs} EPOCHS")
    print("=" * 80)
    
    # Overall training progress bar
    with tqdm(total=training_config.num_epochs, desc="🚀 Training Progress", unit="epoch") as epoch_pbar:
        for epoch in range(1, training_config.num_epochs + 1):
            epoch_pbar.set_description(f"🚀 Epoch {epoch}/{training_config.num_epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Validate
            valid_loss, valid_acc = validate_epoch(model, valid_loader, criterion, device)
            
            # Test (every epoch for real-time plotting)
            test_acc, test_f1, y_true, y_pred, test_loss = evaluate_test(model, test_loader, device)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
            
            # Update progress bar with current metrics
            epoch_pbar.set_postfix({
                'Train Acc': f'{train_acc:.2f}%',
                'Valid Acc': f'{valid_acc:.2f}%', 
                'Test Acc': f'{test_acc*100:.2f}%',
                'LR': f'{current_lr:.6f}'
            })
            
            # Update history
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_accuracy'].append(train_acc)
            history['valid_accuracy'].append(valid_acc)
            history['test_accuracy'].append(test_acc * 100)
            history['test_f1'].append(test_f1)
            history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%, Test F1: {test_f1:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Real-time plotting every epoch
            if epoch % 1 == 0:  # Plot every epoch
                plot_training_curves(history, run_dir, run_number, save_plot=True)
                print(f"📊 Real-time plot updated for epoch {epoch}")
            
            # Save model
            is_best = valid_acc > best_valid_acc
            if is_best:
                best_valid_acc = valid_acc
                patience_counter = 0
                print(f"🎯 NEW BEST MODEL! Validation accuracy: {valid_acc:.4f}")
            else:
                patience_counter += 1
            
            save_model(
                model, optimizer, epoch, train_loss, valid_loss, valid_acc, test_f1,
                run_dir, model_config, training_config, experiment_config, is_best
            )
            
            # Update progress bar
            epoch_pbar.update(1)
            
            # Early stopping
            if training_config.early_stopping and patience_counter >= training_config.early_stopping_patience:
                print(f"\n⚠️ Early stopping triggered after {patience_counter} epochs without improvement")
                break
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    with tqdm(total=1, desc="🧪 Final Testing", unit="phase") as final_pbar:
        test_acc, test_f1, y_true, y_pred, test_loss = evaluate_test(model, test_loader, device)
        final_pbar.update(1)
    
    training_time = time.time() - start_time
    
    # Final metrics
    final_metrics = {
        'final_train_loss': history['train_loss'][-1],
        'final_valid_loss': history['valid_loss'][-1],
        'final_train_accuracy': history['train_accuracy'][-1],
        'final_valid_accuracy': history['valid_accuracy'][-1],
        'final_test_accuracy': test_acc * 100,
        'final_test_f1': test_f1,
        'best_valid_accuracy': best_valid_acc,
        'training_time_seconds': training_time
    }
    
    # Save training history
    save_training_history(history, run_dir, run_number)
    
    # Save experiment summary
    save_experiment_summary(
        run_dir, run_number, final_metrics, model_config, 
        training_config, experiment_config, training_time
    )
    
    # Create plots
    print("\nCreating plots...")
    with tqdm(total=3, desc="📊 Creating visualizations", unit="plot") as plot_pbar:
        plots = create_all_plots(history, y_true, y_pred, run_dir, run_number)
        plot_pbar.update(3)
    
    # Print save information
    print_save_info(run_dir, run_number, final_metrics)
    
    return model, history, final_metrics, y_true, y_pred


def main():
    """Main function to run training."""
    try:
        # Get next run number
        results_dir = Path("result")
        run_number = get_next_run_number(results_dir)
        run_dir = create_run_directory(results_dir, run_number)
        
        print("=" * 80)
        print(f"PARALLEL BRANCH NETWORK TRAINING - RUN {run_number}")
        print("STRUCTURED CONNECTIVITY + MULTIPLE OPTIMIZER SUPPORT")
        print("=" * 80)
        
        # Run training
        model, history, final_metrics, y_true, y_pred = train_model(
            DEFAULT_MODEL_CONFIG,
            DEFAULT_TRAINING_CONFIG,
            DEFAULT_DATA_CONFIG,
            DEFAULT_EXPERIMENT_CONFIG,
            run_dir,
            run_number
        )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved in: {run_dir}")
        print(f"🔢 Run number: {run_number}")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user. Attempting to save latest state...")
        # Best-effort: save an interrupt marker
        interrupt_marker = Path("result") / "INTERRUPTED.txt"
        interrupt_marker.write_text("Training was interrupted by user via KeyboardInterrupt.")
        print(f"Saved interrupt marker at {interrupt_marker}")


if __name__ == "__main__":
    main()