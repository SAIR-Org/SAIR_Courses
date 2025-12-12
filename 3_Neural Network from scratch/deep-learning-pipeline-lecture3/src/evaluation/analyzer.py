"""
Experiment analysis and insights
"""
import numpy as np
from typing import Dict, Any, List, Tuple

from src.utils.logger import PipelineLogger

logger = PipelineLogger.get_logger(__name__)


class ExperimentAnalyzer:
    """Analyze experiment results"""
    
    def __init__(self, config: Dict):
        """
        Initialize analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def calculate_dataset_statistics(self, all_results: Dict) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each dataset
        
        Args:
            all_results: All experiment results
            
        Returns:
            Dataset statistics
        """
        dataset_stats = {}
        
        for dataset in ['mnist', 'fashion', 'cifar10']:
            dataset_accs = [results['test_accuracy'] for (arch, ds), results in all_results.items() if ds == dataset]
            
            if dataset_accs:
                dataset_stats[dataset] = {
                    'mean_accuracy': np.mean(dataset_accs),
                    'std_accuracy': np.std(dataset_accs),
                    'best_accuracy': np.max(dataset_accs),
                    'worst_accuracy': np.min(dataset_accs),
                    'num_models': len(dataset_accs)
                }
        
        return dataset_stats
    
    def get_best_models(self, all_models: Dict, all_results: Dict) -> Dict[str, Dict]:
        """
        Get best model for each dataset
        
        Args:
            all_models: All trained models
            all_results: All experiment results
            
        Returns:
            Best models per dataset
        """
        best_models = {}
        
        for dataset in ['mnist', 'fashion', 'cifar10']:
            # Find results for this dataset
            dataset_results = {k: v for k, v in all_results.items() if k[1] == dataset}
            
            if dataset_results:
                # Get best model
                best_key = max(dataset_results, key=lambda k: dataset_results[k]['test_accuracy'])
                best_model_name = f"{best_key[0]}_{best_key[1]}"
                
                best_models[dataset] = {
                    'name': best_model_name,
                    'model': all_models[best_model_name],
                    'accuracy': dataset_results[best_key]['test_accuracy'],
                    'architecture': best_key[0]
                }
        
        return best_models
    
    def generate_insights(self, dataset_stats: Dict, all_results: Dict) -> List[Tuple[str, str]]:
        """
        Generate insights from experiment
        
        Args:
            dataset_stats: Dataset statistics
            all_results: All experiment results
            
        Returns:
            List of (title, content) insights
        """
        insights = []
        
        # Performance drop with complexity
        if 'mnist' in dataset_stats and 'fashion' in dataset_stats and 'cifar10' in dataset_stats:
            insights.append((
                "📉 Performance Drop with Complexity",
                f"MNIST: {dataset_stats['mnist']['mean_accuracy']:.1%} avg → "
                f"Fashion MNIST: {dataset_stats['fashion']['mean_accuracy']:.1%} avg → "
                f"CIFAR-10: {dataset_stats['cifar10']['mean_accuracy']:.1%} avg\n"
                "   DNNs struggle as data complexity increases."
            ))
        
        # Parameter efficiency
        params_by_arch = {}
        for (arch, dataset), results in all_results.items():
            if arch not in params_by_arch:
                params_by_arch[arch] = []
            params_by_arch[arch].append(results['parameters'])
        
        if len(params_by_arch) > 1:
            avg_params = {arch: np.mean(vals) for arch, vals in params_by_arch.items()}
            insights.append((
                "🏗️ Diminishing Returns with Depth",
                f"Deep DNN has {avg_params.get('deep_dnn', 0)/avg_params.get('simple_dnn', 1):.1f}x more parameters\n"
                "   but only slightly better accuracy → poor parameter efficiency."
            ))
        
        # RGB information loss
        insights.append((
            "🎨 RGB → Grayscale Information Loss",
            "CIFAR-10 converted to grayscale loses critical color information.\n"
            "   A red car vs blue car looks identical to DNN."
        ))
        
        # Translation sensitivity
        insights.append((
            "🔄 Translation Sensitivity",
            "A shirt in top-left vs bottom-right appears completely different.\n"
            "   DNNs have no built-in translation invariance."
        ))
        
        # Parameter explosion
        cifar10_params = [results['parameters'] for (arch, ds), results in all_results.items() if ds == 'cifar10']
        mnist_params = [results['parameters'] for (arch, ds), results in all_results.items() if ds == 'mnist']
        
        if cifar10_params and mnist_params:
            avg_cifar10_params = np.mean(cifar10_params)
            avg_mnist_params = np.mean(mnist_params)
            insights.append((
                "📊 Parameter Explosion",
                f"CIFAR-10: ~{avg_cifar10_params:,.0f} params vs MNIST: ~{avg_mnist_params:,.0f} params\n"
                "   Yet accuracy is lower with more parameters!"
            ))
        
        # Why CNNs are needed
        insights.append((
            "💡 Why CNNs Are Needed",
            "• Parameter sharing → efficiency\n"
            "• Translation invariance → robustness\n"
            "• Hierarchical features → better learning\n"
            "• Spatial preservation → understand relationships"
        ))
        
        return insights
    
    def print_insights(self, insights: List[Tuple[str, str]]) -> None:
        """
        Print insights
        
        Args:
            insights: List of insights
        """
        logger.info("\n🔍 EXPERIMENT INSIGHTS")
        logger.info("=" * 80)
        
        for i, (title, content) in enumerate(insights, 1):
            logger.info(f"\n{i}. {title}")
            for line in content.split('\n'):
                logger.info(f"   {line}")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 CONCLUSION: DNNs work for simple patterns but fail dramatically")
        logger.info("          for complex spatial data. This clearly demonstrates why")
        logger.info("          Convolutional Neural Networks (CNNs) were invented.")
        logger.info("=" * 80)