#!/usr/bin/env python3
"""
Download datasets
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import PipelineLogger
from src.data.loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    
    parser.add_argument("--datasets", nargs="+", choices=["mnist", "fashion", "cifar10"], default=["mnist", "fashion", "cifar10"])
    parser.add_argument("--config", default="config/config.yaml")
    
    args = parser.parse_args()
    
    # Load config
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # Setup logging
    logger = PipelineLogger.setup_logger(
        name="download_data",
        log_level="INFO",
        console=True
    )
    
    logger.info(f"Downloading datasets: {args.datasets}")
    
    # Create data loader
    data_loader = DataLoader(config)
    
    # Download datasets
    for dataset in args.datasets:
        try:
            logger.info(f"\nDownloading {dataset.upper()}...")
            data_loader.load_dataset(dataset)
        except Exception as e:
            logger.error(f"Failed to download {dataset}: {e}")
    
    logger.info("\n✅ Download complete!")


if __name__ == "__main__":
    main()