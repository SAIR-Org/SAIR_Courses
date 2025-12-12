#!/usr/bin/env python3
"""
Command-line tools for the Deep Learning Pipeline
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import PipelineLogger
from src.data.loader import DataLoader
from src.data.preprocesser import DataPreprocessor
from src.training.trainer import ModelTrainer
from src.training.visualizer import TrainingVisualizer
from src.evaluation.analyzer import ExperimentAnalyzer
from src.inference.ui import ModelInferenceUI


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Deep Learning Pipeline CLI Tools")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download datasets')
    download_parser.add_argument("--datasets", nargs="+", choices=["mnist", "fashion", "cifar10"], 
                               help="Datasets to download")
    download_parser.add_argument("--config", type=str, default="config/config.yaml",
                               help="Path to configuration file")
    download_parser.add_argument("--log-level", type=str, default="INFO",
                               choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a specific model')
    train_parser.add_argument("--dataset", required=True, choices=["mnist", "fashion", "cifar10"],
                            help="Dataset to train on")
    train_parser.add_argument("--architecture", required=True, 
                            choices=["simple_dnn", "medium_dnn", "deep_dnn"],
                            help="Architecture to train")
    train_parser.add_argument("--config", type=str, default="config/config.yaml",
                            help="Path to configuration file")
    train_parser.add_argument("--epochs", type=int, help="Override default epochs")
    train_parser.add_argument("--batch-size", type=int, help="Override default batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Override default learning rate")
    train_parser.add_argument("--l2-lambda", type=float, help="Override default L2 regularization")
    train_parser.add_argument("--log-level", type=str, default="INFO",
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch inference UI')
    ui_parser.add_argument("--config", type=str, default="config/config.yaml",
                          help="Path to configuration file")
    ui_parser.add_argument("--datasets", nargs="+", choices=["mnist", "fashion", "cifar10"],
                          help="Datasets to load models for")
    ui_parser.add_argument("--architectures", nargs="+", 
                          choices=["simple_dnn", "medium_dnn", "deep_dnn"],
                          help="Architectures to load")
    ui_parser.add_argument("--share-ui", action="store_true",
                          help="Create public URL for UI")
    ui_parser.add_argument("--log-level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate existing models')
    eval_parser.add_argument("--config", type=str, default="config/config.yaml",
                           help="Path to configuration file")
    eval_parser.add_argument("--datasets", nargs="+", choices=["mnist", "fashion", "cifar10"],
                           help="Datasets to evaluate")
    eval_parser.add_argument("--architectures", nargs="+", 
                           choices=["simple_dnn", "medium_dnn", "deep_dnn"],
                           help="Architectures to evaluate")
    eval_parser.add_argument("--log-level", type=str, default="INFO",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'download':
        return download_data(args)
    elif args.command == 'train':
        return train_model(args)
    elif args.command == 'ui':
        return launch_ui(args)
    elif args.command == 'evaluate':
        return evaluate_models(args)


def download_data(args):
    """Download datasets only"""
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # Setup logging
    logger = PipelineLogger.setup_logger(
        name="download",
        log_level=args.log_level,
        log_file=config['logging'].get('file'),
        console=True
    )
    
    logger.info("📥 Downloading datasets...")
    
    # Determine which datasets to download
    datasets = args.datasets or config['datasets']['enabled']
    
    # Create data loader and download
    data_loader = DataLoader(config)
    
    for dataset in datasets:
        logger.info(f"Downloading {dataset}...")
        try:
            data_loader.load_dataset(dataset, download=True)
            logger.info(f"✅ {dataset} downloaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset}: {e}")
    
    logger.info("✅ All datasets downloaded!")
    return 0


def train_model(args):
    """Train a specific model"""
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # Setup logging
    logger = PipelineLogger.setup_logger(
        name="train",
        log_level=args.log_level,
        log_file=config['logging'].get('file'),
        console=True
    )
    
    logger.info(f"🏗️ Training {args.architecture} on {args.dataset}...")
    
    # Override config if specified
    if args.epochs:
        config['models']['architectures'][args.architecture]['epochs'] = args.epochs
    if args.batch_size:
        config['models']['architectures'][args.architecture]['batch_size'] = args.batch_size
    if args.learning_rate:
        config['models']['architectures'][args.architecture]['learning_rate'] = args.learning_rate
    if args.l2_lambda:
        config['models']['architectures'][args.architecture]['l2_lambda'] = args.l2_lambda
    
    # Create directories
    config_loader.create_directories()
    
    # Initialize components
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config)
    
    # Load and preprocess data
    logger.info(f"Loading {args.dataset} data...")
    raw_data = data_loader.load_all_datasets([args.dataset])
    prepared_data = preprocessor.prepare_all_datasets(raw_data)
    
    # Train the specific model
    logger.info(f"Training {args.architecture} on {args.dataset}...")
    all_models, all_results = trainer.run_experiment_matrix(
        prepared_data=prepared_data,
        architectures=[args.architecture],
        datasets=[args.dataset]
    )
    
    # Save training summary
    from pathlib import Path
    results_dir = Path(config['directories']['results'])
    trainer.save_training_summary(all_results, results_dir / f"{args.architecture}_{args.dataset}_summary.json")
    
    logger.info("✅ Training completed!")
    return 0


def launch_ui(args):
    """Launch the inference UI"""
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # Setup logging
    logger = PipelineLogger.setup_logger(
        name="ui",
        log_level=args.log_level,
        log_file=config['logging'].get('file'),
        console=True
    )
    
    logger.info("🚀 Launching Inference UI...")
    
    # Create directories
    config_loader.create_directories()
    
    # Initialize components
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config)
    analyzer = ExperimentAnalyzer(config)
    
    # Determine which models to load
    architectures = args.architectures or list(config['models']['architectures'].keys())
    datasets = args.datasets or config['datasets']['enabled']
    
    # Load models
    logger.info(f"Loading models for: {datasets}")
    all_models, all_results = trainer.load_trained_models(architectures, datasets)
    
    if not all_models:
        logger.error("❌ No trained models found!")
        logger.error("Please run training first with: dl-train or dl-pipeline --run-mode train")
        return 1
    
    logger.info(f"✅ Loaded {len(all_models)} models")
    
    # Load data for UI
    logger.info("Loading datasets for UI...")
    raw_datasets = data_loader.load_all_datasets(datasets)
    prepared_data = preprocessor.prepare_all_datasets(raw_datasets)
    
    # Get best models - handle missing test_accuracy key
    best_models = {}
    if all_results:
        try:
            # Prepare results for analyzer - ensure test_accuracy exists
            prepared_results = {}
            for key, result in all_results.items():
                prepared_result = result.copy()
                
                # Extract test_accuracy from test_metrics if needed
                if 'test_accuracy' not in prepared_result and 'test_metrics' in prepared_result:
                    if 'accuracy' in prepared_result['test_metrics']:
                        prepared_result['test_accuracy'] = prepared_result['test_metrics']['accuracy']
                
                prepared_results[key] = prepared_result
            
            best_models = analyzer.get_best_models(all_models, prepared_results)
        except Exception as e:
            logger.warning(f"Could not determine best models: {e}")
            best_models = {}
    
    # Launch UI
    ui = ModelInferenceUI(config, all_models, prepared_data, all_results, best_models)
    ui.launch(share=args.share_ui)
    
    return 0


def evaluate_models(args):
    """Evaluate existing models"""
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # Setup logging
    logger = PipelineLogger.setup_logger(
        name="evaluate",
        log_level=args.log_level,
        log_file=config['logging'].get('file'),
        console=True
    )
    
    logger.info("📊 Evaluating models...")
    
    # Create directories
    config_loader.create_directories()
    
    # Initialize components
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config)
    visualizer = TrainingVisualizer(config)
    analyzer = ExperimentAnalyzer(config)
    
    # Determine which models to load
    architectures = args.architectures or list(config['models']['architectures'].keys())
    datasets = args.datasets or config['datasets']['enabled']
    
    # Load models
    logger.info(f"Loading models...")
    all_models, all_results = trainer.load_trained_models(architectures, datasets)
    
    if not all_results:
        logger.error("❌ No trained models found to evaluate!")
        logger.error("Please run training first with: dl-train or dl-pipeline --run-mode train")
        return 1
    
    logger.info(f"✅ Loaded {len(all_results)} model results")
    
    # Load data for evaluation
    logger.info("Loading datasets for evaluation...")
    raw_datasets = data_loader.load_all_datasets(datasets)
    prepared_data = preprocessor.prepare_all_datasets(raw_datasets)
    
    # Compute test metrics if they're None
    logger.info("Computing test metrics...")
    for (arch, dataset), results in all_results.items():
        if all_models and (arch, dataset) in all_models:
            model = all_models[(arch, dataset)]
            
            if ('test_metrics' in results and 
                results['test_metrics']['accuracy'] is None and
                prepared_data and dataset in prepared_data):
                
                try:
                    data_info = prepared_data[dataset]
                    X_test = data_info['data']['X_test']
                    y_test = data_info['data']['y_test']
                    
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    test_loss, test_accuracy = model.evaluate(X_test_flat, y_test)
                    
                    results['test_metrics']['accuracy'] = test_accuracy
                    results['test_metrics']['loss'] = test_loss
                    results['test_accuracy'] = test_accuracy
                    
                    logger.debug(f"Computed metrics for {arch}_{dataset}: {test_accuracy:.2%}")
                except Exception as e:
                    logger.error(f"Failed to compute metrics for {arch}_{dataset}: {e}")
    
    # Prepare results for visualizer and analyzer
    prepared_results = {}
    for key, result in all_results.items():
        prepared_result = result.copy()
        
        # Ensure test_accuracy exists for visualizer/analyzer
        if 'test_accuracy' not in prepared_result and 'test_metrics' in prepared_result:
            if 'accuracy' in prepared_result['test_metrics']:
                prepared_result['test_accuracy'] = prepared_result['test_metrics']['accuracy']
        
        prepared_results[key] = prepared_result
    
    # Generate visualizations
    if config['evaluation'].get('save_visualizations', True):
        logger.info("Generating training history plots...")
        for (arch, dataset), results in prepared_results.items():
            model_name = f"{arch}_{dataset}"
            
            if 'history' in results:
                plot_results = results.copy()
                history_data = plot_results['history']
                
                if isinstance(history_data, dict) and 'train_loss' in history_data:
                    for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch_times']:
                        if key in history_data:
                            plot_results[key] = history_data[key]
                
                try:
                    visualizer.plot_training_history(
                        results=plot_results,
                        model_name=model_name,
                        dataset_name=dataset.upper(),
                        save=True
                    )
                except Exception as e:
                    logger.error(f"Failed to plot history for {model_name}: {e}")
    
    # Plot experiment comparison
    if config['evaluation'].get('comparison_plots', True):
        logger.info("Generating comparison plots...")
        try:
            visualizer.plot_experiment_comparison(prepared_results, save=True)
        except Exception as e:
            logger.error(f"Failed to create comparison plots: {e}")
    
    # Generate insights
    try:
        logger.info("Generating insights...")
        dataset_stats = analyzer.calculate_dataset_statistics(prepared_results)
        insights = analyzer.generate_insights(dataset_stats, prepared_results)
        analyzer.print_insights(insights)
    except Exception as e:
        logger.error(f"Failed to generate insights: {e}")
    
    # Get best models
    if all_models:
        try:
            best_models = analyzer.get_best_models(all_models, prepared_results)
            
            logger.info("\n🏆 BEST MODELS PER DATASET")
            for dataset, model_info in best_models.items():
                logger.info(f"   {dataset.upper():<12} → {model_info['name']:<25} ({model_info['accuracy']:.2%})")
        except Exception as e:
            logger.error(f"Failed to get best models: {e}")
    
    logger.info("✅ Evaluation completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
