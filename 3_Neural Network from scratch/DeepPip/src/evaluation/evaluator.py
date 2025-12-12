# =============================================================================
# File: src/evaluation/evaluator.py
# =============================================================================

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.model.dnn import DNN
from src.data_pipeline.loader import DataLoader
from src.data_pipeline.preprocessor import DataPreprocessor


class Evaluator:
    """Model evaluation and comparison."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(self, model, X_test, y_test, dataset_name):
        """Evaluate single model."""
        loss, accuracy = model.evaluate(X_test, y_test)
        
        results = {
            'model_name': model.name,
            'dataset': dataset_name,
            'test_loss': float(loss),
            'test_accuracy': float(accuracy),
            'parameters': int(model.num_params),
            'history': {
                k: [float(x) for x in v] 
                for k, v in model.history.items()
            }
        }
        
        # Save results
        results_file = self.results_dir / f"{model.name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot training curves
        self._plot_training_history(model, dataset_name)
        
        return results
    
    def _plot_training_history(self, model, dataset_name):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(model.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, model.history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, model.history['val_loss'], 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, model.history['train_acc'], 'b-', label='Train')
        axes[1].plot(epochs, model.history['val_acc'], 'r-', label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model.name} on {dataset_name.upper()}')
        plt.tight_layout()
        
        plot_file = self.results_dir / f"{model.name}_training.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plot saved: {plot_file}")
    
    def compare_all_models(self):
        """Compare all trained models."""
        models_dir = Path(self.config['models_dir'])
        results_files = list(self.results_dir.glob('*_results.json'))
        
        if not results_files:
            self.logger.warning("No results found. Train models first.")
            return
        
        # Load all results
        all_results = []
        for results_file in results_files:
            with open(results_file, 'r') as f:
                all_results.append(json.load(f))
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        model_names = [r['model_name'] for r in all_results]
        datasets = [r['dataset'] for r in all_results]
        accuracies = [r['test_accuracy'] for r in all_results]
        parameters = [r['parameters'] for r in all_results]
        
        # Color by dataset
        dataset_colors = {'mnist': 'blue', 'fashion': 'orange', 'cifar10': 'green'}
        colors = [dataset_colors[d] for d in datasets]
        
        # 1. Accuracy comparison
        x_pos = np.arange(len(model_names))
        bars = axes[0].bar(x_pos, accuracies, color=colors, edgecolor='black')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_title('Model Comparison')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. Parameters vs Accuracy
        axes[1].scatter(parameters, accuracies, s=100, c=colors, 
                       edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Parameters')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Parameter Efficiency')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Dataset performance
        dataset_accs = {}
        for dataset in ['mnist', 'fashion', 'cifar10']:
            dataset_accs[dataset] = np.mean([
                r['test_accuracy'] for r in all_results if r['dataset'] == dataset
            ])
        
        axes[2].bar(range(len(dataset_accs)), list(dataset_accs.values()),
                   color=[dataset_colors[d] for d in dataset_accs.keys()],
                   edgecolor='black')
        axes[2].set_ylabel('Average Accuracy')
        axes[2].set_title('Dataset Difficulty')
        axes[2].set_xticks(range(len(dataset_accs)))
        axes[2].set_xticklabels([d.upper() for d in dataset_accs.keys()])
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Model Comparison Across Datasets and Architectures')
        plt.tight_layout()
        
        comparison_file = self.results_dir / 'comparison.png'
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison plot saved: {comparison_file}")
        
        # Print summary table
        self.logger.info("\n" + "=" * 85)
        self.logger.info("MODEL COMPARISON SUMMARY")
        self.logger.info("=" * 85)
        self.logger.info(f"{'Model':<25} {'Dataset':<12} {'Accuracy':<12} {'Parameters':<12}")
        self.logger.info("-" * 85)
        
        for r in sorted(all_results, key=lambda x: x['test_accuracy'], reverse=True):
            self.logger.info(
                f"{r['model_name']:<25} {r['dataset']:<12} "
                f"{r['test_accuracy']:<12.2%} {r['parameters']:<12,}"
            )
        
        self.logger.info("=" * 85)