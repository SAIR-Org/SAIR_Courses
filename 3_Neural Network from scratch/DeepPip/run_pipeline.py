
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logger
from src.data_pipeline.loader import DataLoader
from src.data_pipeline.preprocessor import DataPreprocessor
from src.model.dnn import DNN
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
import yaml 


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_data_pipeline(config, logger):
    """Load and preprocess all datasets."""
    logger.info("=" * 80)
    logger.info("STEP 1: Loading and preprocessing datasets")
    logger.info("=" * 80)
    
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    
    prepared_data = {}
    for dataset_name in ['mnist', 'fashion', 'cifar10']:
        logger.info(f"\nProcessing {dataset_name.upper()}...")
        
        # Load
        train_data, test_data, classes = data_loader.load_dataset(dataset_name)
        
        # Preprocess
        data_dict = preprocessor.prepare_dataset(
            *train_data, *test_data, dataset_name
        )
        
        prepared_data[dataset_name] = {
            'data': data_dict,
            'classes': classes
        }
        
        logger.info(f"  Train: {data_dict['X_train'].shape}")
        logger.info(f"  Val: {data_dict['X_val'].shape}")
        logger.info(f"  Test: {data_dict['X_test'].shape}")
    
    return prepared_data


def run_training_pipeline(config, logger, dataset, arch, prepared_data=None):
    """Train a specific model."""
    logger.info("=" * 80)
    logger.info(f"STEP 2: Training {arch} on {dataset}")
    logger.info("=" * 80)
    
    # Load data if not provided
    if prepared_data is None:
        data_loader = DataLoader(config)
        preprocessor = DataPreprocessor(config)
        train_data, test_data, classes = data_loader.load_dataset(dataset)
        data_dict = preprocessor.prepare_dataset(*train_data, *test_data, dataset)
    else:
        data_dict = prepared_data[dataset]['data']
    
    # Build model
    arch_config = config['architectures'][arch]
    input_size = data_dict['X_train'].shape[1]
    
    # Replace null with input size
    layers = [input_size if x is None else x for x in arch_config['layers']]
    
    model = DNN(
        layer_sizes=layers,
        activations=arch_config['activations'],
        learning_rate=arch_config['learning_rate'],
        l2_lambda=arch_config['l2_lambda'],
        name=f"{arch}_{dataset}"
    )
    
    # Train
    trainer = Trainer(config, logger)
    trainer.train(
        model=model,
        X_train=data_dict['X_train'],
        y_train=data_dict['y_train'],
        X_val=data_dict['X_val'],
        y_val=data_dict['y_val'],
        epochs=arch_config['epochs'],
        batch_size=arch_config['batch_size']
    )
    
    # Save
    model_path = Path(config['models_dir']) / f"{arch}_{dataset}.pkl"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    return model


def run_evaluation_pipeline(config, logger, dataset, arch):
    """Evaluate a trained model."""
    logger.info("=" * 80)
    logger.info(f"STEP 3: Evaluating {arch} on {dataset}")
    logger.info("=" * 80)
    
    # Load model
    model_path = Path(config['models_dir']) / f"{arch}_{dataset}.pkl"
    model = DNN.load(str(model_path))
    
    # Load test data
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    train_data, test_data, classes = data_loader.load_dataset(dataset)
    data_dict = preprocessor.prepare_dataset(*train_data, *test_data, dataset)
    
    # Evaluate
    evaluator = Evaluator(config, logger)
    results = evaluator.evaluate(
        model=model,
        X_test=data_dict['X_test'],
        y_test=data_dict['y_test'],
        dataset_name=dataset
    )
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy: {results['test_accuracy']:.2%}")
    logger.info(f"  Loss: {results['test_loss']:.4f}")
    
    return results


def run_comparison_pipeline(config, logger):
    """Compare all trained models."""
    logger.info("=" * 80)
    logger.info("STEP 4: Comparing all models")
    logger.info("=" * 80)
    
    evaluator = Evaluator(config, logger)
    evaluator.compare_all_models()

def run_all_pipeline(config, logger):
    """Run complete pipeline: data, training, evaluation, comparison."""
    logger.info("=" * 80)
    logger.info("RUNNING COMPLETE PIPELINE")
    logger.info("=" * 80)
    
    # 1. Load data
    prepared_data = run_data_pipeline(config, logger)
    
    # 2. Train and evaluate all models
    all_results = {}
    for arch in ['simple_dnn', 'medium_dnn', 'deep_dnn']:
        for dataset in ['mnist', 'fashion', 'cifar10']:
            # Train
            model = run_training_pipeline(config, logger, dataset, arch, prepared_data)
            
            # Evaluate immediately
            results = run_evaluation_pipeline(config, logger, dataset, arch)
            all_results[f"{arch}_{dataset}"] = results
    
    # 3. Compare models using evaluation results
    run_comparison_pipeline(config, logger)
    
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)



def main():
    parser = argparse.ArgumentParser(
        description="DNN Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --mode all
  
  # Train specific model
  python run_pipeline.py --mode train --dataset mnist --arch simple_dnn
  
  # Evaluate model
  python run_pipeline.py --mode evaluate --dataset mnist --arch simple_dnn
  
  # Compare all models
  python run_pipeline.py --mode compare
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['all', 'data', 'train', 'evaluate', 'compare'],
        required=True,
        help='Pipeline mode'
    )
    parser.add_argument(
        '--dataset',
        choices=['mnist', 'fashion', 'cifar10'],
        help='Dataset name (required for train/evaluate modes)'
    )
    parser.add_argument(
        '--arch',
        choices=['simple_dnn', 'medium_dnn', 'deep_dnn'],
        help='Architecture name (required for train/evaluate modes)'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['train', 'evaluate']:
        if not args.dataset or not args.arch:
            parser.error(f"--dataset and --arch required for mode '{args.mode}'")
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger(
        name='dnn_pipeline',
        log_dir=config['logs_dir'],
        level=args.log_level
    )
    
    # Create directories
    for dir_path in [config['data_dir'], config['models_dir'], 
                     config['results_dir'], config['logs_dir']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    try:
        if args.mode == 'all':
            run_all_pipeline(config, logger)
        elif args.mode == 'data':
            run_data_pipeline(config, logger)
        elif args.mode == 'train':
            run_training_pipeline(config, logger, args.dataset, args.arch)
        elif args.mode == 'evaluate':
            run_evaluation_pipeline(config, logger, args.dataset, args.arch)
        elif args.mode == 'compare':
            run_comparison_pipeline(config, logger)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()