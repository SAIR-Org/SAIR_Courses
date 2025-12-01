"""
Main pipeline runner for Spaceship Titanic ML project.

This script orchestrates the entire ML pipeline from data loading to model deployment.

Usage:
    python run_pipeline.py --mode full
    python run_pipeline.py --mode preprocessing
    python run_pipeline.py --mode training
    python run_pipeline.py --mode evaluation
    python run_pipeline.py --mode submission
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add current directory to Python path to enable imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# FIXED IMPORTS - Remove "Pipeline." prefix since modules are directly in config/, data/, models/
from config.config import Config
from data.load_data import load_train_test_data, prepare_train_val_test_split
from data.preprocessing import preprocess_data, create_preprocessing_pipeline
from models.train_model import train_all_models, select_best_model, generate_submission
from models.evaluate_model import comprehensive_evaluation

import joblib
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Create necessary directories and setup environment."""
    logger.info("Setting up environment...")
    Config.create_directories()
    logger.info("✅ Environment setup complete")


def run_preprocessing(save_processed: bool = True):
    """
    Run data loading and preprocessing pipeline.
    
    Args:
        save_processed: Whether to save processed data to disk
        
    Returns:
        Tuple of processed data and pipeline
    """
    logger.info("="*80)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*80)
    
    # Load data
    logger.info("Loading training and test data...")
    train_df, test_df = load_train_test_data()
    
    # Split data
    logger.info("Splitting data into train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = \
        prepare_train_val_test_split(train_df)
    
    # Create and apply preprocessing pipeline
    logger.info("Creating preprocessing pipeline...")
    pipeline = create_preprocessing_pipeline(X_train)
    
    logger.info("Applying preprocessing...")
    X_train_proc, X_val_proc, X_test_proc, pipeline = preprocess_data(
        X_train, X_val, X_test, pipeline
    )
    
    # Save processed data if requested
    if save_processed:
        logger.info("Saving processed data...")
        processed_dir = Config.PROCESSED_DATA_DIR
        
        np.save(processed_dir / 'X_train_processed.npy', X_train_proc)
        np.save(processed_dir / 'X_val_processed.npy', X_val_proc)
        np.save(processed_dir / 'X_test_processed.npy', X_test_proc)
        np.save(processed_dir / 'y_train.npy', y_train)
        np.save(processed_dir / 'y_val.npy', y_val)
        np.save(processed_dir / 'y_test.npy', y_test)
        
        joblib.dump(pipeline, processed_dir / 'preprocessing_pipeline.pkl')
        
        logger.info(f"✅ Processed data saved to {processed_dir}")
    
    return (X_train_proc, X_val_proc, X_test_proc, 
            y_train, y_val, y_test, 
            pipeline, test_df)


def run_training(X_train_proc, y_train, X_val_proc, y_val):
    """
    Run model training pipeline.
    
    Args:
        X_train_proc: Processed training features
        y_train: Training labels
        X_val_proc: Processed validation features
        y_val: Validation labels
        
    Returns:
        Tuple of results and trained models
    """
    logger.info("="*80)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("="*80)
    
    # Train all models
    results, trained_models = train_all_models(
        X_train_proc, y_train,
        X_val_proc, y_val,
        track_mlflow=True
    )
    
    # Select best model
    best_name, best_model, best_metrics = select_best_model(
        results, trained_models
    )
    
    logger.info(f"\n🏆 Best Model: {best_name}")
    logger.info(f"   Validation Accuracy: {best_metrics['val_accuracy']:.4f}")
    logger.info(f"   CV Accuracy: {best_metrics['cv_accuracy_mean']:.4f} ± {best_metrics['cv_accuracy_std']:.4f}")
    
    return results, trained_models, best_name, best_model


def run_evaluation(model, X_test_proc, y_test, model_name):
    """
    Run comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test_proc: Processed test features
        y_test: Test labels
        model_name: Name of the model
    """
    logger.info("="*80)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("="*80)
    
    metrics = comprehensive_evaluation(
        model, X_test_proc, y_test, model_name
    )
    
    logger.info("\n📊 Test Set Performance:")
    logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"   Precision: {metrics['precision']:.4f}")
    logger.info(f"   Recall: {metrics['recall']:.4f}")
    logger.info(f"   F1-Score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics


def run_submission_generation(model, pipeline, test_df):
    """
    Generate Kaggle submission file.
    
    Args:
        model: Trained model
        pipeline: Preprocessing pipeline
        test_df: Test dataframe
    """
    logger.info("="*80)
    logger.info("STEP 4: SUBMISSION GENERATION")
    logger.info("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = Config.SUBMISSION_DIR / f"submission_{timestamp}.csv"
    
    submission_df = generate_submission(
        model, pipeline, test_df, str(submission_path)
    )
    
    logger.info(f"✅ Submission saved to {submission_path}")
    
    return submission_df


def save_production_model(model, pipeline, model_name, metrics):
    """
    Save model for production use.
    
    Args:
        model: Trained model
        pipeline: Preprocessing pipeline
        model_name: Name of the model
        metrics: Performance metrics
    """
    logger.info("="*80)
    logger.info("SAVING PRODUCTION MODEL")
    logger.info("="*80)
    
    prod_dir = Config.PRODUCTION_MODEL_DIR
    prod_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and pipeline
    model_path = prod_dir / 'final_model.pkl'
    pipeline_path = prod_dir / 'preprocessing_pipeline.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)
    
    # Save model card
    model_card = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in metrics.items()},
        'config': {
            'random_state': Config.RANDOM_STATE,
            'cv_folds': Config.CV_FOLDS
        }
    }
    
    card_path = prod_dir / 'model_card.json'
    with open(card_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    
    logger.info(f"✅ Production model saved to {prod_dir}")
    logger.info(f"   • Model: {model_path}")
    logger.info(f"   • Pipeline: {pipeline_path}")
    logger.info(f"   • Model Card: {card_path}")


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(
        description="Run Spaceship Titanic ML Pipeline"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'preprocessing', 'training', 'evaluation', 'submission'],
        default='full',
        help='Pipeline mode to run'
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 SPACESHIP TITANIC ML PIPELINE")
    logger.info("="*80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Setup environment
    setup_environment()
    
    try:
        if args.mode in ['full', 'preprocessing']:
            # Run preprocessing
            (X_train_proc, X_val_proc, X_test_proc,
             y_train, y_val, y_test,
             pipeline, test_df) = run_preprocessing(save_processed=True)
        
        if args.mode == 'preprocessing':
            logger.info("\n✅ Preprocessing complete!")
            return
        
        if args.mode in ['full', 'training']:
            # Run training
            results, trained_models, best_name, best_model = run_training(
                X_train_proc, y_train, X_val_proc, y_val
            )
        
        if args.mode == 'training':
            logger.info("\n✅ Training complete!")
            return
        
        if args.mode in ['full', 'evaluation']:
            # Retrain on full training data (train + val)
            logger.info("\nRetraining best model on full training data...")
            X_full_train = np.vstack([X_train_proc, X_val_proc])
            y_full_train = np.concatenate([y_train, y_val])
            best_model.fit(X_full_train, y_full_train)
            
            # Run evaluation
            test_metrics = run_evaluation(
                best_model, X_test_proc, y_test, best_name
            )
        
        if args.mode == 'evaluation':
            logger.info("\n✅ Evaluation complete!")
            return
        
        if args.mode in ['full', 'submission']:
            # Generate submission
            submission_df = run_submission_generation(
                best_model, pipeline, test_df
            )
        
        if args.mode in ['full']:
            # Save production model
            save_production_model(
                best_model, pipeline, best_name, test_metrics
            )
        
        logger.info("\n" + "="*80)
        logger.info("🎉 PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info("\nNext steps:")
        logger.info("1. Submit the CSV to Kaggle")
        logger.info("2. View experiments: mlflow ui --backend-store-uri spaceship_experiments")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())