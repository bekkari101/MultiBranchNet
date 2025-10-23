"""
Analyze Hyperparameter Search Results - Extract Top Performing Models
======================================================================
This script analyzes the final_summary.json and extracts the best models
based on test accuracy, saving their complete configurations.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def load_results(results_file: str = 'final_summary.json') -> Dict:
    """Load the hyperparameter search results."""
    results_path = Path(results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded results from: {results_path}")
    print(f"📊 Total runs: {data.get('total_combinations', 'N/A')}")
    print(f"✅ Successful runs: {data.get('successful_runs', 'N/A')}")
    print(f"❌ Failed runs: {data.get('failed_runs', 'N/A')}")
    return data


def extract_top_models(results: Dict, top_n: int = 10) -> List[Dict[str, Any]]:
    """Extract the top N models based on test accuracy."""
    all_results = results.get('all_results', [])
    
    # Filter out failed runs and extract successful results
    successful_results = []
    for result in all_results:
        if 'error' not in result and 'final_metrics' in result:
            test_acc = result['final_metrics'].get('final_test_accuracy', 0)
            successful_results.append({
                'run_id': result['run_id'],
                'test_accuracy': test_acc,
                'model_config': result['model_config'],
                'training_config': result['training_config'],
                'final_metrics': result['final_metrics'],
                'training_time_seconds': result.get('training_time_seconds', 0),
                'timestamp': result.get('timestamp', '')
            })
    
    # Sort by test accuracy (descending)
    successful_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    print(f"\n📈 Found {len(successful_results)} successful runs")
    if successful_results:
        print(f"🏆 Best test accuracy: {successful_results[0]['test_accuracy']:.2f}%")
        print(f"📉 Worst test accuracy: {successful_results[-1]['test_accuracy']:.2f}%")
        print(f"📊 Average test accuracy: {sum(r['test_accuracy'] for r in successful_results) / len(successful_results):.2f}%")
    
    return successful_results[:top_n]


def format_model_summary(model: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Format a model's information for readable output."""
    model_config = model['model_config']
    training_config = model['training_config']
    final_metrics = model['final_metrics']
    
    # Format branch layers
    branch_info = []
    for layer in model_config.get('branch_layers', []):
        branch_info.append(f"{layer['num_branches']}x{layer['nodes_per_branch']}")
    branch_str = " → ".join(branch_info)
    
    # Format conv layers
    conv_channels = model_config.get('conv_channels', [])
    conv_kernel_sizes = model_config.get('conv_kernel_sizes', [])
    conv_str = f"{len(conv_channels)}L: {conv_channels}"
    
    summary = {
        'rank': rank,
        'run_id': model['run_id'],
        'test_accuracy': round(model['test_accuracy'], 2),
        'test_f1_score': round(final_metrics.get('final_test_f1', 0), 4),
        'best_valid_accuracy': round(final_metrics.get('best_valid_accuracy', 0), 2),
        'architecture': {
            'total_parameters': model_config.get('param_count', 0),
            'conv_layers': {
                'num_layers': model_config.get('num_conv_layers', 0),
                'channels': conv_channels,
                'kernel_sizes': conv_kernel_sizes,
                'description': conv_str
            },
            'branch_layers': {
                'num_layers': len(model_config.get('branch_layers', [])),
                'structure': model_config.get('branch_layers', []),
                'description': branch_str
            }
        },
        'training': {
            'optimizer': training_config.get('optimizer', 'N/A'),
            'learning_rate': training_config.get('learning_rate', 0),
            'batch_size': training_config.get('batch_size', 0),
            'num_epochs': training_config.get('num_epochs', 0),
            'actual_epochs': training_config.get('actual_epochs', 0),
            'extended_training': training_config.get('extended_training', False),
            'training_time_seconds': round(model.get('training_time_seconds', 0), 2)
        },
        'metrics': {
            'test_accuracy': round(model['test_accuracy'], 2),
            'test_f1': round(final_metrics.get('final_test_f1', 0), 4),
            'test_loss': round(final_metrics.get('final_test_loss', 0), 4),
            'best_validation_accuracy': round(final_metrics.get('best_valid_accuracy', 0), 2)
        },
        'timestamp': model.get('timestamp', '')
    }
    
    return summary


def save_top_models(top_models: List[Dict[str, Any]], output_dir: str = './hyperparameter_search_results'):
    """Save top models to JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Format all models
    formatted_models = []
    for rank, model in enumerate(top_models, 1):
        formatted_models.append(format_model_summary(model, rank))
    
    # Save complete top models file
    top_models_file = output_path / 'top_models.json'
    with open(top_models_file, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'total_top_models': len(formatted_models),
            'models': formatted_models
        }, f, indent=2)
    
    print(f"\n💾 Saved top models to: {top_models_file}")
    
    # Save individual model files
    individual_dir = output_path / 'top_models_individual'
    individual_dir.mkdir(exist_ok=True)
    
    for model in formatted_models:
        model_file = individual_dir / f"rank_{model['rank']}_run_{model['run_id']}_acc_{model['test_accuracy']:.2f}.json"
        with open(model_file, 'w') as f:
            json.dump(model, f, indent=2)
    
    print(f"💾 Saved individual model files to: {individual_dir}")
    
    # Create a readable summary file
    summary_file = output_path / 'top_models_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TOP PERFORMING MODELS - HYPERPARAMETER SEARCH RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for model in formatted_models:
            f.write(f"{'='*80}\n")
            f.write(f"RANK #{model['rank']} - Run ID: {model['run_id']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Test Accuracy: {model['test_accuracy']:.2f}%\n")
            f.write(f"Test F1 Score: {model['test_f1_score']:.4f}\n")
            f.write(f"Best Validation Accuracy: {model['best_valid_accuracy']:.2f}%\n")
            f.write(f"\nARCHITECTURE:\n")
            f.write(f"   - Total Parameters: {model['architecture']['total_parameters']:,}\n")
            f.write(f"   - Conv Layers: {model['architecture']['conv_layers']['description']}\n")
            f.write(f"   - Branch Layers: {model['architecture']['branch_layers']['description']}\n")
            f.write(f"\nTRAINING CONFIG:\n")
            f.write(f"   - Optimizer: {model['training']['optimizer'].upper()}\n")
            f.write(f"   - Learning Rate: {model['training']['learning_rate']}\n")
            f.write(f"   - Batch Size: {model['training']['batch_size']}\n")
            f.write(f"   - Epochs: {model['training']['actual_epochs']}/{model['training']['num_epochs']}\n")
            if model['training']['extended_training']:
                f.write(f"   - ** Extended Training Applied (98%+ accuracy) **\n")
            f.write(f"   - Training Time: {model['training']['training_time_seconds']:.2f}s\n")
            f.write(f"\nTimestamp: {model['timestamp']}\n")
            f.write(f"\n")
    
    print(f"📄 Saved readable summary to: {summary_file}")
    
    return formatted_models


def print_top_models_table(top_models: List[Dict[str, Any]]):
    """Print a formatted table of top models."""
    print("\n" + "=" * 150)
    print("TOP PERFORMING MODELS")
    print("=" * 150)
    print(f"{'Rank':<6} {'Run':<6} {'Test Acc':<10} {'F1':<8} {'Params':<8} {'Conv':<20} {'Branch':<25} {'Opt':<6} {'LR':<8} {'BS':<6} {'Extended':<10}")
    print("-" * 150)
    
    for rank, model in enumerate(top_models, 1):
        summary = format_model_summary(model, rank)
        
        conv_desc = f"{summary['architecture']['conv_layers']['num_layers']}L{summary['architecture']['conv_layers']['channels']}"
        branch_desc = summary['architecture']['branch_layers']['description']
        
        extended = "✅ Yes" if summary['training']['extended_training'] else "No"
        
        print(f"{rank:<6} "
              f"{summary['run_id']:<6} "
              f"{summary['test_accuracy']:<10.2f} "
              f"{summary['test_f1_score']:<8.4f} "
              f"{summary['architecture']['total_parameters']:<8,} "
              f"{conv_desc:<20} "
              f"{branch_desc[:24]:<25} "
              f"{summary['training']['optimizer'].upper():<6} "
              f"{summary['training']['learning_rate']:<8} "
              f"{summary['training']['batch_size']:<6} "
              f"{extended:<10}")
    
    print("=" * 150)


def analyze_patterns(top_models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze patterns in top performing models."""
    if not top_models:
        return {}
    
    patterns = {
        'optimizers': {},
        'learning_rates': {},
        'batch_sizes': {},
        'conv_layers': {},
        'branch_layers': {},
        'extended_training_count': 0,
        'avg_parameters': 0,
        'avg_test_accuracy': 0,
        'avg_training_time': 0
    }
    
    total_params = 0
    total_acc = 0
    total_time = 0
    
    for model in top_models:
        summary = format_model_summary(model, 0)
        
        # Count optimizers
        opt = summary['training']['optimizer']
        patterns['optimizers'][opt] = patterns['optimizers'].get(opt, 0) + 1
        
        # Count learning rates
        lr = summary['training']['learning_rate']
        patterns['learning_rates'][lr] = patterns['learning_rates'].get(lr, 0) + 1
        
        # Count batch sizes
        bs = summary['training']['batch_size']
        patterns['batch_sizes'][bs] = patterns['batch_sizes'].get(bs, 0) + 1
        
        # Count conv layers
        conv = summary['architecture']['conv_layers']['num_layers']
        patterns['conv_layers'][conv] = patterns['conv_layers'].get(conv, 0) + 1
        
        # Count branch layers
        branch = summary['architecture']['branch_layers']['num_layers']
        patterns['branch_layers'][branch] = patterns['branch_layers'].get(branch, 0) + 1
        
        # Extended training
        if summary['training']['extended_training']:
            patterns['extended_training_count'] += 1
        
        # Accumulate stats
        total_params += summary['architecture']['total_parameters']
        total_acc += summary['test_accuracy']
        total_time += summary['training']['training_time_seconds']
    
    n = len(top_models)
    patterns['avg_parameters'] = total_params // n
    patterns['avg_test_accuracy'] = round(total_acc / n, 2)
    patterns['avg_training_time'] = round(total_time / n, 2)
    
    return patterns


def print_patterns(patterns: Dict[str, Any]):
    """Print analysis of patterns in top models."""
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS - TOP MODELS")
    print("=" * 80)
    
    print("\n📊 Optimizer Distribution:")
    for opt, count in sorted(patterns['optimizers'].items(), key=lambda x: x[1], reverse=True):
        print(f"   • {opt.upper()}: {count} models")
    
    print("\n📊 Learning Rate Distribution:")
    for lr, count in sorted(patterns['learning_rates'].items(), key=lambda x: x[1], reverse=True):
        print(f"   • {lr}: {count} models")
    
    print("\n📊 Batch Size Distribution:")
    for bs, count in sorted(patterns['batch_sizes'].items(), key=lambda x: x[1], reverse=True):
        print(f"   • {bs}: {count} models")
    
    print("\n📊 Conv Layers Distribution:")
    for conv, count in sorted(patterns['conv_layers'].items(), key=lambda x: x[1], reverse=True):
        print(f"   • {conv} layers: {count} models")
    
    print("\n📊 Branch Layers Distribution:")
    for branch, count in sorted(patterns['branch_layers'].items(), key=lambda x: x[1], reverse=True):
        print(f"   • {branch} layers: {count} models")
    
    print(f"\n⭐ Extended Training:")
    print(f"   • {patterns['extended_training_count']} models received extended training (98%+ accuracy)")
    
    print(f"\n📈 Average Statistics:")
    print(f"   • Average Parameters: {patterns['avg_parameters']:,}")
    print(f"   • Average Test Accuracy: {patterns['avg_test_accuracy']:.2f}%")
    print(f"   • Average Training Time: {patterns['avg_training_time']:.2f}s")
    
    print("=" * 80)


def main():
    """Main execution function."""
    print("=" * 80)
    print("ANALYZING HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)
    
    # Configuration
    results_file = 'final_summary.json'
    top_n = 20  # Number of top models to extract
    
    try:
        # Load results
        results = load_results(results_file)
        
        # Extract top models
        top_models = extract_top_models(results, top_n=top_n)
        
        if not top_models:
            print("\n❌ No successful models found!")
            return
        
        # Print table
        print_top_models_table(top_models)
        
        # Analyze patterns
        patterns = analyze_patterns(top_models)
        print_patterns(patterns)
        
        # Save results
        formatted_models = save_top_models(top_models)
        
        print("\n✅ Analysis complete!")
        print(f"📁 Results saved in: ./hyperparameter_search_results/")
        print(f"   • top_models.json - Complete structured data")
        print(f"   • top_models_summary.txt - Readable summary")
        print(f"   • top_models_individual/ - Individual JSON files for each model")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure the hyperparameter search has been completed.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
