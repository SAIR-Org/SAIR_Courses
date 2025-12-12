"""
Training visualization
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.utils.logger import PipelineLogger

logger = PipelineLogger.get_logger(__name__)


class TrainingVisualizer:
    """Visualize training results"""
    
    def __init__(self, config: Dict):
        """
        Initialize visualizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.viz_dir = Path(config['directories']['visualizations'])
        self.viz_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, results: Dict, model_name: str, 
                             dataset_name: str, save: bool = True) -> None:
        """
        Plot training history
        
        Args:
            results: Training results (either full results dict or history dict)
            model_name: Name of model
            dataset_name: Name of dataset
            save: Whether to save plot
        """
        # Extract history from results - handle both structures
        if 'history' in results:
            history = results['history']
        elif 'train_loss' in results:
            history = results  # Results already contain history keys
        else:
            logger.error(f"Cannot find history data in results for {model_name}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 1. Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Accuracy curves
        ax = axes[0, 1]
        ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        if 'val_acc' in history and history['val_acc']:
            ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Training time per epoch
        ax = axes[1, 0]
        if 'epoch_times' in history and history['epoch_times']:
            ax.plot(epochs, history['epoch_times'], 'purple', marker='o', alpha=0.7)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Training Time per Epoch')
            ax.grid(True, alpha=0.3)
        
        # 4. Loss-Accuracy tradeoff
        ax = axes[1, 1]
        scatter = ax.scatter(history['train_loss'], history['train_acc'], 
                           c=epochs, cmap='viridis', s=50, alpha=0.7)
        ax.set_xlabel('Loss')
        ax.set_ylabel('Accuracy')
        ax.set_title('Loss vs Accuracy (Epochs)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Epoch')
        
        plt.suptitle(f'{model_name} on {dataset_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            # Clean model name for filename
            clean_model_name = model_name.replace(' ', '_').replace('/', '_')
            clean_dataset_name = dataset_name.replace(' ', '_')
            save_path = self.viz_dir / f"training_{clean_model_name}_{clean_dataset_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training plot: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_experiment_comparison(self, all_results: Dict, save: bool = True) -> None:
        """
        Plot experiment comparison
        
        Args:
            all_results: All experiment results
            save: Whether to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Extract data
        model_names = []
        datasets = []
        test_accuracies = []
        parameters = []
        
        for (model_name, dataset_name), results in all_results.items():
            model_names.append(model_name)
            datasets.append(dataset_name)
            test_accuracies.append(results['test_accuracy'])
            parameters.append(results['num_parameters'])  # Changed from 'parameters' to 'num_parameters'
        
        # Color by dataset
        dataset_colors = {'mnist': 'blue', 'fashion_mnist': 'orange', 'fashion': 'orange', 'cifar10': 'green'}
        colors = [dataset_colors.get(d, 'gray') for d in datasets]
        
        # 1. Test accuracy comparison
        x_pos = np.arange(len(model_names))
        bars = axes[0].bar(x_pos, test_accuracies, color=colors, edgecolor='black')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Test Accuracy Comparison (9 Models)')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f"{m}\n({d})" for m, d in zip(model_names, datasets)], 
                               rotation=45, ha='right', fontsize=8)
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add accuracy labels
        for bar, acc in zip(bars, test_accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.1%}', ha='center', va='bottom', fontsize=8)
        
        # 2. Parameters vs Accuracy
        scatter = axes[1].scatter(parameters, test_accuracies, s=100, c=colors, 
                                 edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Number of Parameters')
        axes[1].set_ylabel('Test Accuracy')
        axes[1].set_title('Parameters vs Accuracy Efficiency')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        
        # 3. Dataset performance comparison
        ax = axes[2]
        dataset_means = {}
        unique_datasets = set(datasets)
        for dataset in unique_datasets:
            dataset_accs = [acc for d, acc in zip(datasets, test_accuracies) if d == dataset]
            if dataset_accs:
                dataset_means[dataset] = np.mean(dataset_accs)
        
        bars_ds = ax.bar(range(len(dataset_means)), list(dataset_means.values()), 
                        color=[dataset_colors.get(d, 'gray') for d in dataset_means.keys()], 
                        edgecolor='black')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Average Test Accuracy')
        ax.set_title('Dataset Difficulty Comparison')
        ax.set_xticks(range(len(dataset_means)))
        ax.set_xticklabels([f"{d.upper()}\n({len([x for x in datasets if x == d])} models)" 
                           for d in dataset_means.keys()])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, acc in zip(bars_ds, dataset_means.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.1%}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('3×3 Experiment Matrix: 3 Architectures × 3 Datasets', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.viz_dir / 'experiment_comparison.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved experiment comparison: {save_path}")
        
        plt.show()
        plt.close()
        
        # Print summary table
        print("\n📊 EXPERIMENT SUMMARY")
        print("=" * 85)
        print(f"{'Model':<20} {'Dataset':<15} {'Test Acc':<12} {'Params':<15} {'Params/Acc':<15}")
        print("-" * 85)
        
        for (model_name, dataset_name), results in all_results.items():
            params = results['num_parameters']  # Changed from 'parameters'
            acc = results['test_accuracy']
            efficiency = params / (acc + 1e-8)
            
            print(f"{model_name:<20} {dataset_name:<15} {acc:<12.2%} {params:<15,} {efficiency:,.0f}")
        
        print("-" * 85)