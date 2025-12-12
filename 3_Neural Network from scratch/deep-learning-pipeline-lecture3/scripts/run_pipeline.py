#!/usr/bin/env python3
"""
Main pipeline runner script
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Deep Learning Pipeline")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="config/experiments/lecture3_experiment.yaml",
        help="Path to experiment configuration"
    )
    
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["all", "data", "train", "evaluate", "ui"],
        default="all",
        help="Pipeline run mode"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["mnist", "fashion", "cifar10"],
        help="Datasets to process (default: all)"
    )
    
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=["simple_dnn", "medium_dnn", "deep_dnn"],
        help="Architectures to train (default: all)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Don't launch UI even if configured"
    )
    
    parser.add_argument(
        "--share-ui",
        action="store_true",
        help="Create public URL for UI"
    )
    
    return parser.parse_args()


def main():
    """Main pipeline execution"""
    args = parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # Setup logging
    log_config = config['logging']
    logger = PipelineLogger.setup_logger(
        name="pipeline",
        log_level=args.log_level,
        log_file=log_config.get('file'),
        log_format=log_config.get('format'),
        console=log_config.get('console', True)
    )
    
    logger.info("🚀 Starting Deep Learning Pipeline")
    logger.info(f"Config: {args.config}")
    logger.info(f"Run mode: {args.run_mode}")
    
    # Create directories
    config_loader.create_directories()
    
    # Initialize components
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config)
    visualizer = TrainingVisualizer(config)
    analyzer = ExperimentAnalyzer(config)
    
    # Variables to store results
    raw_datasets = None
    prepared_data = None
    all_models = None
    all_results = None
    best_models = None
    
    # Run pipeline based on mode
    if args.run_mode in ["all", "data"]:
        logger.info("\n📥 STEP 1: Loading Data")
        logger.info("=" * 60)
        
        # Load datasets
        dataset_names = args.datasets or config['datasets']['enabled']
        raw_datasets = data_loader.load_all_datasets(dataset_names)
        
        # Preprocess data
        logger.info("\n🔧 STEP 2: Preprocessing Data")
        logger.info("=" * 60)
        prepared_data = preprocessor.prepare_all_datasets(raw_datasets)
    
    if args.run_mode in ["all", "train"]:
        logger.info("\n🏗️ STEP 3: Training Models")
        logger.info("=" * 60)
        
        # Determine architectures and datasets
        architectures = args.architectures or list(config['models']['architectures'].keys())
        datasets = args.datasets or config['datasets']['enabled']
        
        # Run experiment matrix
        all_models, all_results = trainer.run_experiment_matrix(
            prepared_data=prepared_data,
            architectures=architectures,
            datasets=datasets
        )
        
        # Save training summary
        results_dir = Path(config['directories']['results'])
        trainer.save_training_summary(all_results, results_dir / "training_summary.json")
    
    if args.run_mode in ["all", "evaluate"]:
        logger.info("\n📊 STEP 4: Evaluation & Visualization")
        logger.info("=" * 60)
        
        # Load results if not already available
        if all_results is None:
            logger.info("Loading previously trained models and results...")
            
            # Determine architectures and datasets
            architectures = args.architectures or list(config['models']['architectures'].keys())
            datasets = args.datasets or config['datasets']['enabled']
            
            # Try to load saved models
            all_models, all_results = trainer.load_trained_models(architectures, datasets)
            
            if not all_results:
                logger.error("❌ No trained models found to evaluate!")
                logger.error("Please run training first with: python scripts/run_pipeline.py --run-mode train")
                return
            else:
                logger.info(f"✅ Loaded {len(all_results)} model results")
        
        # Load prepared data for evaluation if needed
        if prepared_data is None and args.run_mode == "evaluate":
            logger.info("Loading datasets for evaluation context...")
            dataset_names = args.datasets or config['datasets']['enabled']
            raw_datasets = data_loader.load_all_datasets(dataset_names)
            prepared_data = preprocessor.prepare_all_datasets(raw_datasets)
        
        # Compute test metrics if they're None
        logger.info("Computing test metrics for loaded models...")
        for (arch, dataset), results in all_results.items():
            if all_models and (arch, dataset) in all_models:
                model = all_models[(arch, dataset)]
                
                # Check if we need to compute test metrics
                if ('test_metrics' in results and 
                    results['test_metrics']['accuracy'] is None and
                    prepared_data and dataset in prepared_data):
                    
                    try:
                        data_info = prepared_data[dataset]
                        X_test = data_info['data']['X_test']
                        y_test = data_info['data']['y_test']
                        
                        # Flatten for DNN
                        X_test_flat = X_test.reshape(X_test.shape[0], -1)
                        
                        # Evaluate
                        test_loss, test_accuracy = model.evaluate(X_test_flat, y_test)
                        
                        # Update results
                        results['test_metrics']['accuracy'] = test_accuracy
                        results['test_metrics']['loss'] = test_loss
                        results['test_accuracy'] = test_accuracy  # Add for compatibility
                        results['test_loss'] = test_loss  # Add for compatibility
                        
                        logger.debug(f"Computed metrics for {arch}_{dataset}: accuracy={test_accuracy:.2%}")
                    except Exception as e:
                        logger.error(f"Failed to compute metrics for {arch}_{dataset}: {e}")
        
        # Visualize training histories
        if config['evaluation'].get('save_visualizations', True):
            logger.info("Generating training history plots...")
            for (arch, dataset), results in all_results.items():
                model_name = f"{arch}_{dataset}"
                
                # Check if we have history data
                if 'history' not in results:
                    logger.warning(f"No history data found for {model_name}")
                    continue
                
                # The visualizer expects 'results' parameter, not 'history'
                # Prepare results for the visualizer
                plot_results = results.copy()
                
                # Copy history data to top level if needed
                history_data = plot_results['history']
                if isinstance(history_data, dict):
                    # Check if history data has the expected structure
                    if 'train_loss' in history_data:
                        # Copy history keys to top level
                        for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch_times']:
                            if key in history_data:
                                plot_results[key] = history_data[key]
                
                try:
                    visualizer.plot_training_history(
                        results=plot_results,  # Correct parameter name
                        model_name=model_name,
                        dataset_name=dataset.upper(),
                        save=True
                    )
                except Exception as e:
                    logger.error(f"Failed to plot history for {model_name}: {e}")
        
        # Plot experiment comparison
        if config['evaluation'].get('comparison_plots', True):
            logger.info("Generating experiment comparison plots...")
            
            # Prepare results for the comparison plot
            plot_results = {}
            for key, result in all_results.items():
                plot_result = result.copy()
                
                # Get test_accuracy
                if 'test_accuracy' not in plot_result:
                    if 'test_metrics' in plot_result and 'accuracy' in plot_result['test_metrics']:
                        plot_result['test_accuracy'] = plot_result['test_metrics']['accuracy']
                    elif 'accuracy' in plot_result:
                        plot_result['test_accuracy'] = plot_result['accuracy']
                
                # Get parameters
                if 'parameters' not in plot_result:
                    if 'num_parameters' in plot_result:
                        plot_result['parameters'] = plot_result['num_parameters']
                    elif 'model_info' in plot_result and 'num_parameters' in plot_result['model_info']:
                        plot_result['parameters'] = plot_result['model_info']['num_parameters']
                
                # Only include if we have the required data
                if 'test_accuracy' in plot_result and 'parameters' in plot_result:
                    plot_results[key] = plot_result
                else:
                    logger.warning(f"Skipping {key} for comparison plot - missing required data")
            
            if plot_results:
                try:
                    visualizer.plot_experiment_comparison(plot_results, save=True)
                except Exception as e:
                    logger.error(f"Failed to create comparison plots: {e}")
                    # Debug
                    if plot_results:
                        first_key = list(plot_results.keys())[0]
                        logger.debug(f"Sample result keys: {list(plot_results[first_key].keys())}")
            else:
                logger.error("No valid results for comparison plot!")
        
        # Generate insights
        try:
            logger.info("Generating insights and analysis...")
            
            # Prepare results for analyzer
            analysis_results = {}
            for key, result in all_results.items():
                analysis_result = result.copy()
                
                # Ensure required keys for analyzer
                if 'test_accuracy' not in analysis_result and 'test_metrics' in analysis_result:
                    if 'accuracy' in analysis_result['test_metrics']:
                        analysis_result['test_accuracy'] = analysis_result['test_metrics']['accuracy']
                
                if 'num_parameters' not in analysis_result and 'model_info' in analysis_result:
                    if 'num_parameters' in analysis_result['model_info']:
                        analysis_result['num_parameters'] = analysis_result['model_info']['num_parameters']
                
                analysis_results[key] = analysis_result
            
            dataset_stats = analyzer.calculate_dataset_statistics(analysis_results)
            insights = analyzer.generate_insights(dataset_stats, analysis_results)
            analyzer.print_insights(insights)
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
        
        # Get best models
        try:
            if all_models:
                # Prepare results for get_best_models
                best_model_results = {}
                for key, result in all_results.items():
                    best_result = result.copy()
                    
                    # Ensure test_accuracy exists
                    if 'test_accuracy' not in best_result and 'test_metrics' in best_result:
                        if 'accuracy' in best_result['test_metrics']:
                            best_result['test_accuracy'] = best_result['test_metrics']['accuracy']
                    
                    best_model_results[key] = best_result
                
                best_models = analyzer.get_best_models(all_models, best_model_results)
                
                logger.info("\n🏆 BEST MODELS PER DATASET")
                for dataset, model_info in best_models.items():
                    logger.info(f"   {dataset.upper():<12} → {model_info['name']:<25} ({model_info['accuracy']:.2%})")
        except Exception as e:
            logger.error(f"Failed to get best models: {e}")
    
    if args.run_mode in ["all", "ui"] and not args.no_ui:
        logger.info("\n🚀 STEP 5: Launching Inference UI")
        logger.info("=" * 60)
        
        # Load models if needed
        if all_models is None:
            logger.info("Loading models for UI...")
            architectures = args.architectures or list(config['models']['architectures'].keys())
            datasets = args.datasets or config['datasets']['enabled']
            all_models, all_results = trainer.load_trained_models(architectures, datasets)
        
        if all_models is None or not all_models:
            logger.error("❌ No trained models found for UI!")
            logger.error("Please run training first with: python scripts/run_pipeline.py --run-mode train")
            return
        
        # Get best models if not already available
        if best_models is None and all_results:
            try:
                best_models = analyzer.get_best_models(all_models, all_results)
            except:
                best_models = {}
        
        # Load data if needed for UI
        if prepared_data is None:
            logger.info("Loading datasets for UI...")
            dataset_names = args.datasets or config['datasets']['enabled']
            raw_datasets = data_loader.load_all_datasets(dataset_names)
            prepared_data = preprocessor.prepare_all_datasets(raw_datasets)
        
        # Launch UI
        ui = ModelInferenceUI(config, all_models, prepared_data, all_results, best_models)
        ui.launch(share=args.share_ui)
    
    logger.info("\n✅ Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()