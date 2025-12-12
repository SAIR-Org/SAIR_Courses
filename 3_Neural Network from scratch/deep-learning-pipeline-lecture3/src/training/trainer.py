# src/training/trainer.py
import numpy as np
from pathlib import Path
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.neural_network import DNN
from src.utils.logger import PipelineLogger

class ModelTrainer:
    """Trainer for DNN models from first principles"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models_config = config['models']
        self.training_config = config['training']
        self.directories = config['directories']
        self.results = {}
        self.logger = PipelineLogger.get_logger("trainer")
        
    def run_experiment_matrix(self, prepared_data, architectures, datasets):
        """Run all combinations of architectures × datasets (9 models)"""
        all_models = {}
        all_results = {}
        
        total_experiments = len(architectures) * len(datasets)
        current_exp = 1
        
        self.logger.info(f"Running {total_experiments} experiments...")
        self.logger.info(f"Architectures: {architectures}")
        self.logger.info(f"Datasets: {datasets}")
        
        for dataset_name in datasets:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"📊 DATASET: {dataset_name.upper()}")
            self.logger.info('='*60)
            
            # Get dataset info
            dataset_info = prepared_data[dataset_name]
            data_dict = dataset_info['data']
            
            # Extract data based on the actual structure
            X_train_raw = data_dict['X_train']
            y_train = data_dict['y_train']
            X_val_raw = data_dict['X_val']
            y_val = data_dict['y_val']
            X_test_raw = data_dict['X_test']
            y_test = data_dict['y_test']
            
            # Get shape information
            input_shape = dataset_info.get('original_shape', X_train_raw.shape[1:])
            num_classes = y_train.shape[1]  # y_train is one-hot encoded
            
            # Flatten data for DNN (if not already flattened)
            if X_train_raw.ndim > 2:
                X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
                X_val = X_val_raw.reshape(X_val_raw.shape[0], -1)
                X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)
            else:
                X_train = X_train_raw
                X_val = X_val_raw
                X_test = X_test_raw
            
            self.logger.info(f"Original input shape: {input_shape}")
            self.logger.info(f"Flattened input size: {X_train.shape[1]}")
            self.logger.info(f"Number of classes: {num_classes}")
            self.logger.info(f"Training samples: {X_train.shape[0]:,}")
            self.logger.info(f"Validation samples: {X_val.shape[0]:,}")
            self.logger.info(f"Test samples: {X_test.shape[0]:,}")
            
            for arch_name in architectures:
                self.logger.info(f"\n🏗️ Architecture: {arch_name}")
                self.logger.info('-'*40)
                
                # Get architecture configuration
                arch_config = self.models_config['architectures'][arch_name]
                
                # Build layer sizes
                layer_sizes = arch_config['layer_sizes'].copy()
                
                # Replace None with actual input size
                if layer_sizes[0] is None:
                    layer_sizes[0] = X_train.shape[1]
                
                # Ensure output size matches num_classes
                layer_sizes[-1] = num_classes
                
                # Build model
                model = DNN(
                    layer_sizes=layer_sizes,
                    activations=arch_config['activations'],
                    learning_rate=arch_config['learning_rate'],
                    l2_lambda=arch_config['l2_lambda'],
                    name=f"{arch_name}_{dataset_name}"
                )
                
                self.logger.info(f"Model: {model.layer_sizes}")
                self.logger.info(f"Activations: {model.activations}")
                self.logger.info(f"Learning rate: {arch_config['learning_rate']}")
                self.logger.info(f"L2 lambda: {arch_config['l2_lambda']}")
                
                # Train model
                start_time = time.time()
                
                model.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=arch_config['epochs'],
                    batch_size=arch_config['batch_size'],
                    verbose=self.training_config.get('verbose', True)
                )
                
                training_time = time.time() - start_time
                
                # Evaluate on test set
                test_loss, test_accuracy = model.evaluate(X_test, y_test)
                
                # Store results
                key = (arch_name, dataset_name)
                all_models[key] = model
                
                # Prepare history with additional metadata
                history_with_metadata = model.history.copy()
                history_with_metadata.update({
                    'dataset': dataset_name,
                    'architecture': arch_name,
                    'training_time': training_time,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'num_parameters': model.num_params,
                    'input_shape': input_shape,
                    'num_classes': num_classes,
                    'config': {
                        'layer_sizes': model.layer_sizes,
                        'activations': model.activations,
                        'learning_rate': arch_config['learning_rate'],
                        'l2_lambda': arch_config['l2_lambda'],
                        'epochs': arch_config['epochs'],
                        'batch_size': arch_config['batch_size']
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
                all_results[key] = history_with_metadata
                
                # Save model
                model_path = self._save_model(model, arch_name, dataset_name)
                self.logger.info(f"💾 Model saved to: {model_path}")
                self.logger.info(f"⏱️  Training time: {training_time:.1f}s")
                self.logger.info(f"🎯 Test accuracy: {test_accuracy:.2%}")
                self.logger.info(f"📈 Parameters: {model.num_params:,}")
                
                current_exp += 1
        
        self.logger.info(f"\n✅ All {total_experiments} experiments completed!")
        
        # Generate summary
        summary = self._generate_experiment_summary(all_results)
        self.logger.info(f"\n📊 EXPERIMENT SUMMARY:")
        self.logger.info("="*60)
        
        for dataset in datasets:
            self.logger.info(f"\n{dataset.upper()}:")
            for arch in architectures:
                key = (arch, dataset)
                if key in all_results:
                    result = all_results[key]
                    self.logger.info(f"  {arch:<12} → Test Acc: {result['test_accuracy']:.2%} "
                                f"(Time: {result['training_time']:.1f}s, Params: {result['num_parameters']:,})")
        
        return all_models, all_results
    
    def _save_model(self, model: DNN, arch_name: str, dataset_name: str) -> Path:
        """Save trained model to disk"""
        models_dir = Path(self.directories['models'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{arch_name}_{dataset_name}_{timestamp}.pkl"
        model_path = models_dir / filename
        
        model.save(model_path)
        
        # Also save as latest
        latest_path = models_dir / f"{arch_name}_{dataset_name}_latest.pkl"
        model.save(latest_path)
        
        return model_path
    
    def _generate_experiment_summary(self, all_results: Dict) -> Dict:
        """Generate comprehensive experiment summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(all_results),
            'experiments': {},
            'statistics': {},
            'best_models': {}
        }
        
        # Group by dataset
        dataset_results = {}
        
        for (arch, dataset), results in all_results.items():
            # Store experiment details
            exp_key = f"{arch}_{dataset}"
            summary['experiments'][exp_key] = {
                'dataset': dataset,
                'architecture': arch,
                'test_accuracy': results['test_accuracy'],
                'test_loss': results['test_loss'],
                'training_time': results['training_time'],
                'num_parameters': results['num_parameters'],
                'final_train_accuracy': results['train_acc'][-1] if results['train_acc'] else 0,
                'final_val_accuracy': results['val_acc'][-1] if results['val_acc'] else 0,
                'epochs_trained': len(results['train_acc']),
                'avg_epoch_time': np.mean(results['epoch_times']) if results['epoch_times'] else 0,
                'input_shape': results.get('input_shape', 'unknown'),
                'num_classes': results.get('num_classes', 'unknown')
            }
            
            # Group by dataset for statistics
            if dataset not in dataset_results:
                dataset_results[dataset] = []
            dataset_results[dataset].append({
                'architecture': arch,
                'accuracy': results['test_accuracy'],
                'training_time': results['training_time'],
                'parameters': results['num_parameters']
            })
        
        # Calculate statistics and find best models
        for dataset, results in dataset_results.items():
            accuracies = [r['accuracy'] for r in results]
            training_times = [r['training_time'] for r in results]
            parameters = [r['parameters'] for r in results]
            
            # Find best model for this dataset
            best_idx = np.argmax(accuracies)
            best_model = results[best_idx]
            
            summary['statistics'][dataset] = {
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'min_accuracy': float(np.min(accuracies)),
                'max_accuracy': float(np.max(accuracies)),
                'mean_training_time': float(np.mean(training_times)),
                'total_parameters_sum': int(np.sum(parameters)),
                'accuracy_range': float(np.max(accuracies) - np.min(accuracies))
            }
            
            summary['best_models'][dataset] = {
                'architecture': best_model['architecture'],
                'accuracy': float(best_model['accuracy']),
                'training_time': float(best_model['training_time']),
                'parameters': int(best_model['parameters'])
            }
        
        return summary
    
    def save_training_summary(self, all_results: Dict, output_path: Path) -> Dict:
        """
        Save comprehensive training summary to JSON file
        
        Args:
            all_results: Dictionary containing all experiment results
            output_path: Path to save the summary JSON
            
        Returns:
            Summary dictionary
        """
        summary = self._generate_experiment_summary(all_results)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                elif isinstance(obj, tuple):
                    return list(obj)
                else:
                    return obj
            
            serializable_summary = convert_to_serializable(summary)
            json.dump(serializable_summary, f, indent=2)
        
        self.logger.info(f"📊 Training summary saved to: {output_path}")
        
        # Print key insights
        self.logger.info(f"\n🔍 KEY INSIGHTS:")
        self.logger.info("="*60)
        
        for dataset, best_model in summary['best_models'].items():
            stats = summary['statistics'][dataset]
            self.logger.info(f"\n{dataset.upper()}:")
            self.logger.info(f"  Best model: {best_model['architecture']} "
                           f"({best_model['accuracy']:.2%})")
            self.logger.info(f"  Accuracy range: {stats['accuracy_range']:.2%}")
            self.logger.info(f"  Mean training time: {stats['mean_training_time']:.1f}s")
            self.logger.info(f"  Total parameters across architectures: {stats['total_parameters_sum']:,}")
        
        return summary
    
    def load_trained_models(self, architectures: List[str], datasets: List[str]) -> Tuple[Dict, Dict]:
        """
        Load previously trained models from disk
        
        Args:
            architectures: List of architecture names
            datasets: List of dataset names
            
        Returns:
            Tuple of (all_models, all_results)
        """
        all_models = {}
        all_results = {}
        models_dir = Path(self.directories['models'])
        
        for dataset in datasets:
            for arch in architectures:
                # Try to find the latest model
                model_pattern = f"{arch}_{dataset}_*.pkl"
                model_files = list(models_dir.glob(model_pattern))
                
                if not model_files:
                    self.logger.warning(f"No trained model found for {arch}_{dataset}")
                    continue
                
                # Get the most recent model
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                
                try:
                    # Load the model
                    model = DNN.load(latest_model)
                    
                    # Create results entry
                    key = (arch, dataset)
                    all_models[key] = model
                    
                    # Create simplified results entry for evaluation
                    all_results[key] = {
                        'dataset': dataset,
                        'architecture': arch,
                        'history': model.history,
                        'test_metrics': {
                            'loss': None,  # Will be computed during evaluation
                            'accuracy': None
                        },
                        'model_info': {
                            'name': model.name,
                            'num_parameters': model.num_params,
                            'layer_sizes': model.layer_sizes,
                            'activations': model.activations
                        }
                    }
                    
                    self.logger.info(f"✅ Loaded {model.name} from {latest_model.name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load model {latest_model}: {e}")
        
        if not all_models:
            self.logger.error("No models could be loaded!")
        else:
            self.logger.info(f"Loaded {len(all_models)} models")
        
        return all_models, all_results
    
    def plot_training_history(self, all_results: Dict, output_dir: Path):
        """
        Plot training history for all experiments
        
        Args:
            all_results: Dictionary containing all experiment results
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
        
        for (arch, dataset), results in all_results.items():
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'{arch} on {dataset}', fontsize=16, fontweight='bold')
            
            # Plot training/validation loss
            epochs = range(1, len(results['train_loss']) + 1)
            
            axes[0, 0].plot(epochs, results['train_loss'], 'b-', label='Training Loss', linewidth=2)
            if 'val_loss' in results and len(results['val_loss']) > 0:
                axes[0, 0].plot(epochs, results['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot training/validation accuracy
            axes[0, 1].plot(epochs, results['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
            if 'val_acc' in results and len(results['val_acc']) > 0:
                axes[0, 1].plot(epochs, results['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot epoch times
            if 'epoch_times' in results and len(results['epoch_times']) > 0:
                axes[1, 0].plot(epochs, results['epoch_times'], 'g-', linewidth=2)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Time (seconds)')
                axes[1, 0].set_title('Epoch Training Times')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Add text box with metrics
            textstr = '\n'.join([
                f'Final Test Accuracy: {results["test_accuracy"]:.2%}',
                f'Final Test Loss: {results["test_loss"]:.4f}',
                f'Training Time: {results["training_time"]:.1f}s',
                f'Parameters: {results["num_parameters"]:,}',
                f'Best Epoch: {np.argmin(results["val_loss"]) + 1 if "val_loss" in results else "N/A"}'
            ])
            
            axes[1, 1].text(0.05, 0.95, textstr, transform=axes[1, 1].transAxes, 
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plot_path = output_dir / f'{arch}_{dataset}_training.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📈 Plot saved: {plot_path}")