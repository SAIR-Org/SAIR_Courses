#!/usr/bin/env python3
"""
Train a single model
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import PipelineLogger
from src.data.loader import DataLoader
from src.data.preprocesser import DataPreprocessor
from src.models.neural_network import DNN


def main():
    parser = argparse.ArgumentParser(description="Train a single model")
    
    parser.add_argument("--dataset", required=True, choices=["mnist", "fashion", "cifar10"])
    parser.add_argument("--architecture", required=True, choices=["simple_dnn", "medium_dnn", "deep_dnn"])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--output", help="Output model path")
    
    args = parser.parse_args()
    
    # Load config
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # Setup logging
    logger = PipelineLogger.setup_logger(
        name="train_single",
        log_level="INFO",
        console=True
    )
    
    logger.info(f"Training {args.architecture} on {args.dataset}")
    
    # Load and preprocess data
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor(config)
    
    raw_data = data_loader.load_dataset(args.dataset)
    prepared = preprocessor.prepare_dataset(*raw_data[:2], dataset_name=args.dataset)
    
    # Get model config
    model_config = config['models']['architectures'][args.architecture].copy()
    
    # Override config with command line args
    if args.epochs:
        model_config['epochs'] = args.epochs
    if args.learning_rate:
        model_config['learning_rate'] = args.learning_rate
    
    # Create model
    input_size = prepared['X_train'].shape[1]
    model_config['layer_sizes'][0] = input_size
    
    model = DNN(
        layer_sizes=model_config['layer_sizes'],
        activations=model_config['activations'],
        learning_rate=model_config['learning_rate'],
        l2_lambda=model_config['l2_lambda'],
        name=f"{args.architecture}_{args.dataset}"
    )
    
    # Train model
    model.train(
        X_train=prepared['X_train'],
        y_train=prepared['y_train'],
        X_val=prepared['X_val'],
        y_val=prepared['y_val'],
        epochs=model_config['epochs'],
        batch_size=args.batch_size
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(prepared['X_test'], prepared['y_test'])
    logger.info(f"Test Accuracy: {test_acc:.2%}, Test Loss: {test_loss:.4f}")
    
    # Save model
    output_path = args.output or f"models/{args.architecture}_{args.dataset}.pkl"
    model.save(Path(output_path))
    
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()