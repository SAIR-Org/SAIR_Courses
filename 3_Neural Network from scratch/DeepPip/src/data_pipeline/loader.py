# =============================================================================
# File: src/data/loader_annotated.py
# Annotated version of the dataset loader with line-by-line explanations
# =============================================================================

import os
import gzip
import struct
import tarfile
import pickle as pkl
import urllib.request
from pathlib import Path
from tqdm import tqdm
import numpy as np

class DataLoader:
    """Load datasets from URLs with caching.

    This class provides a small, explicit pipeline to download, cache,
    extract (if necessary), and parse datasets that are distributed in
    different formats (MNIST-style IDX .gz files and CIFAR-10 tar/pickle
    archive). The comments in this file explain each step in detail.
    """

    def __init__(self, config):
        # Save configuration (URLs, class names, data_dir, etc.)
        self.config = config

        # Base directory where datasets will be stored. Use pathlib for
        # convenient cross-platform path handling. Create the directory
        # if it doesn't already exist (parents=True allows nested dirs).
        self.data_dir = Path(config['data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, dataset_name):
        """Public entry point: choose the appropriate loader by name.

        We dispatch to different private methods depending on the dataset
        because MNIST-style datasets and CIFAR-10 are encoded differently
        on disk.
        """
        if dataset_name in ['mnist', 'fashion']:
            # MNIST and Fashion-MNIST use the same IDX format; handle
            # them with a shared loader.
            return self._load_mnist_style(dataset_name)
        elif dataset_name == 'cifar10':
            # CIFAR-10 is packaged as a tar.gz containing pickled batches.
            return self._load_cifar10()
        else:
            # If the caller requests an unknown dataset, raise early.
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _download_file(self, url, filepath, max_retries=3):
        """Download file with progress bar and basic retry logic.

        - url: remote URL to download from
        - filepath: local Path to write bytes to
        - max_retries: try a few times before failing

        Uses urllib.request directly (no external downloader). The tqdm
        progress bar wraps the write loop so the user sees download
        progress. If the download fails, we retry up to max_retries.
        """
        for attempt in range(max_retries):
            try:
                # Open the URL. This returns a file-like response object.
                with urllib.request.urlopen(url) as response:
                    # Try to read the Content-Length header for the progress bar.
                    total = int(response.headers.get("Content-Length", 0))

                    # Open local file for binary writing and create a tqdm bar.
                    with open(filepath, "wb") as f, tqdm(
                        total=total, unit="B", unit_scale=True,
                        desc=filepath.name
                    ) as bar:
                        # Read the response in chunks. This is memory-efficient
                        # and allows the progress bar to update.
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            bar.update(len(chunk))
                # If we reach here, download succeeded — return True.
                return True
            except Exception as e:
                # On failure, either retry (if attempts remain) or re-raise.
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries}: {e}")
                else:
                    # No retries left: surface the original exception.
                    raise
        # If we exit the loop without returning, the download failed.
        return False

    def _load_mnist_style(self, dataset_name):
        """Load datasets that follow the IDX format (MNIST / Fashion-MNIST).

        Summary of format (images file):
        - 4 bytes: magic number (2051 for images)
        - 4 bytes: number of images
        - 4 bytes: number of rows
        - 4 bytes: number of columns
        - remaining bytes: pixels (unsigned byte per pixel)

        Summary of labels file:
        - 4 bytes: magic number (2049 for labels)
        - 4 bytes: number of labels
        - remaining bytes: label bytes (unsigned byte per label)
        """
        # Load dataset-specific configuration (URLs and classes) from config.
        dataset_config = self.config['datasets'][dataset_name]

        # Directory for this dataset (e.g., data/mnist/)
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # --- Download files if missing ---
        files = {}
        for file_type, url in dataset_config['urls'].items():
            # Save each remote file as <file_type>.gz in the dataset dir.
            filepath = dataset_dir / f"{file_type}.gz"
            if not filepath.exists():
                # Only download when missing — this enables caching.
                print(f"Downloading {file_type}...")
                self._download_file(url, filepath)
            # Store resolved local path for later reading.
            files[file_type] = filepath

        # --- Helper: parse image IDX gzip file ---
        def load_images(path):
            # The file is gzip-compressed; open with gzip in binary mode.
            with gzip.open(path, 'rb') as f:
                # The header is 16 bytes: four big-endian unsigned ints.
                # '>IIII' means big-endian (>) and four unsigned ints (I).
                magic, num, rows, cols = struct.unpack('>IIII', f.read(16))

                # Read the rest of the file (raw pixel bytes). Use
                # np.frombuffer to create a 1-D uint8 view of the bytes
                # and then reshape into (num, rows, cols).
                images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
            return images

        # --- Helper: parse label IDX gzip file ---
        def load_labels(path):
            with gzip.open(path, 'rb') as f:
                # Labels header is 8 bytes: two unsigned ints.
                magic, num = struct.unpack('>II', f.read(8))

                # Remaining bytes are the labels themselves (one uint8 each).
                labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

        # Load train/test images and labels using helpers above.
        X_train = load_images(files['train_images'])
        y_train = load_labels(files['train_labels'])
        X_test = load_images(files['test_images'])
        y_test = load_labels(files['test_labels'])

        # The config may include a human-readable list of class names.
        classes = dataset_config['classes']

        # Return tuples consistent with common dataset APIs: (X, y), (X_test, y_test), classes
        return (X_train, y_train), (X_test, y_test), classes

    def _load_cifar10(self):
        """Load CIFAR-10 which is distributed as a tar.gz archive containing
        multiple Python-pickled batch files.

        Archive structure after extraction (cifar-10-batches-py/):
            data_batch_1
            data_batch_2
            data_batch_3
            data_batch_4
            data_batch_5
            test_batch
            batches.meta

        Each data_batch is a pickled dict with keys: 'data', 'labels', 'filenames'.
        - 'data' is a (10000, 3072) uint8 array: 3072 = 32*32*3
        - We reshape into (N, 3, 32, 32) then transpose to (N, 32, 32, 3)
          to produce standard HWC image format.
        """
        dataset_config = self.config['datasets']['cifar10']

        # Create dataset directory if necessary
        dataset_dir = self.data_dir / 'cifar10'
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Local paths for the archive and the extracted folder
        tar_path = dataset_dir / 'cifar-10-python.tar.gz'
        extracted_dir = dataset_dir / 'cifar-10-batches-py'

        # Download the tar.gz archive if missing
        if not tar_path.exists():
            print("Downloading CIFAR-10...")
            self._download_file(dataset_config['urls']['archive'], tar_path)

        # Extract the tarball if the extracted folder is not already present
        if not extracted_dir.exists():
            print("Extracting CIFAR-10...")
            # Use tarfile to extract safely into dataset_dir
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=dataset_dir)

        # --- Helper: load a single pickled batch file ---
        def load_batch(path):
            # Open the batch file in binary mode and unpickle it.
            # encoding='latin1' is required because the CIFAR pickles were
            # created with Python2; latin1 preserves raw byte values.
            with open(path, 'rb') as f:
                batch = pkl.load(f, encoding='latin1')

                # 'data' is shape (N, 3072). First reshape to (N, 3, 32, 32)
                # because CIFAR stores channels-first inside the flattened
                # buffer, then transpose to (N, 32, 32, 3) for HWC ordering.
                X = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

                # Convert labels list to a numpy array for consistency.
                y = np.array(batch['labels'])
            return X, y

        # --- Load and concatenate the five training batches ---
        X_train_list, y_train_list = [], []
        for i in range(1, 6):
            batch_path = extracted_dir / f'data_batch_{i}'
            X_batch, y_batch = load_batch(batch_path)
            X_train_list.append(X_batch)
            y_train_list.append(y_batch)

        # Concatenate lists of arrays into single training arrays.
        # After concatenation, X_train shape will be (50000, 32, 32, 3).
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Load the single test batch (10000 images)
        X_test, y_test = load_batch(extracted_dir / 'test_batch')

        # Load class names from config (if present)
        classes = dataset_config['classes']

        return (X_train, y_train), (X_test, y_test), classes
