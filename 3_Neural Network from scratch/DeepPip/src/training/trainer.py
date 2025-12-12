# =============================================================================
# File: src/training/trainer.py
# =============================================================================

import time
import numpy as np
from tqdm import tqdm


class Trainer:
    """Model training orchestrator."""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def train(self, model, X_train, y_train, X_val, y_val, epochs, batch_size):
        """Train model with progress tracking."""
        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        self.logger.info(f"Training {model.name}")
        self.logger.info(f"  Parameters: {model.num_params:,}")
        self.logger.info(f"  Samples: {n_samples:,}")
        self.logger.info(f"  Batches: {n_batches}")
        self.logger.info(f"  Epochs: {epochs}")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss, epoch_acc = 0, 0
            
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            with tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{epochs}", 
                     leave=False, disable=self.logger.level > 20) as pbar:
                for batch in range(n_batches):
                    start = batch * batch_size
                    end = min(start + batch_size, n_samples)
                    
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]
                    
                    loss, acc = model.train_step(X_batch, y_batch)
                    epoch_loss += loss
                    epoch_acc += acc
                    
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.2%}'})
                    pbar.update(1)
            
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            # Validation
            val_loss, val_acc = model.evaluate(X_val, y_val)
            
            # Record
            model.history['train_loss'].append(epoch_loss)
            model.history['train_acc'].append(epoch_acc)
            model.history['val_loss'].append(val_loss)
            model.history['val_acc'].append(val_acc)
            
            # Log
            elapsed = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: loss={epoch_loss:.4f} acc={epoch_acc:.2%} | "
                f"Val: loss={val_loss:.4f} acc={val_acc:.2%} | "
                f"Time: {elapsed:.1f}s"
            )
        
        self.logger.info(f"Training complete for {model.name}")
