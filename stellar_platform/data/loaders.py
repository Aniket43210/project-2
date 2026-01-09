"""Data loaders for stellar classification models.

Provides TensorFlow and PyTorch data loaders for efficient batching,
preprocessing, and augmentation of astronomical data.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, List, Callable, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SpectralDataLoader:
    """Data loader for spectral classification."""
    
    def __init__(
        self,
        spectra: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False,
        normalize: bool = True
    ):
        """Initialize spectral data loader.
        
        Args:
            spectra: Array of shape (n_samples, n_wavelengths)
            labels: Array of shape (n_samples,) or (n_samples, n_classes)
            batch_size: Batch size for training
            shuffle: Whether to shuffle data each epoch
            augment: Whether to apply data augmentation
            normalize: Whether to normalize spectra
        """
        self.spectra = spectra.astype(np.float32)
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.normalize = normalize
        
        self.n_samples = len(spectra)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        self.indices = np.arange(self.n_samples)
        
        # Compute normalization statistics
        if normalize:
            self.mean = np.mean(spectra, axis=0, keepdims=True)
            self.std = np.std(spectra, axis=0, keepdims=True) + 1e-8
        
        self.reset()
    
    def reset(self):
        """Reset iterator for new epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_batch = 0
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.n_batches
    
    def __iter__(self):
        """Make iterable."""
        self.reset()
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch."""
        if self.current_batch >= self.n_batches:
            raise StopIteration
        
        # Get batch indices
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch data
        batch_spectra = self.spectra[batch_indices].copy()
        batch_labels = self.labels[batch_indices]
        
        # Normalize
        if self.normalize:
            batch_spectra = (batch_spectra - self.mean) / self.std
        
        # Augment
        if self.augment:
            batch_spectra = self._augment_spectra(batch_spectra)
        
        self.current_batch += 1
        return batch_spectra, batch_labels
    
    def _augment_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Apply data augmentation to spectra.
        
        Args:
            spectra: Batch of spectra (batch_size, n_wavelengths)
        
        Returns:
            Augmented spectra
        """
        # Add Gaussian noise
        noise_level = 0.02
        noise = np.random.randn(*spectra.shape) * noise_level
        spectra = spectra + noise
        
        # Random flux scaling (simulating calibration uncertainty)
        scale = np.random.uniform(0.95, 1.05, (spectra.shape[0], 1))
        spectra = spectra * scale
        
        # Random baseline shift
        shift = np.random.uniform(-0.01, 0.01, (spectra.shape[0], 1))
        spectra = spectra + shift
        
        return spectra


class LightCurveDataLoader:
    """Data loader for light curve classification."""
    
    def __init__(
        self,
        lightcurves: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False,
        max_length: Optional[int] = None
    ):
        """Initialize light curve data loader.
        
        Args:
            lightcurves: Array of shape (n_samples, max_timesteps) or list of variable-length arrays
            labels: Array of shape (n_samples,) or (n_samples, n_classes)
            batch_size: Batch size for training
            shuffle: Whether to shuffle data each epoch
            augment: Whether to apply data augmentation
            max_length: Maximum sequence length (pad/truncate)
        """
        self.lightcurves = lightcurves
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.max_length = max_length
        
        self.n_samples = len(lightcurves)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        self.indices = np.arange(self.n_samples)
        
        self.reset()
    
    def reset(self):
        """Reset iterator for new epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_batch = 0
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.n_batches
    
    def __iter__(self):
        """Make iterable."""
        self.reset()
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get next batch.
        
        Returns:
            Tuple of (lightcurves, labels, mask) where mask indicates valid timesteps
        """
        if self.current_batch >= self.n_batches:
            raise StopIteration
        
        # Get batch indices
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch data
        batch_lcs = [self.lightcurves[i].copy() for i in batch_indices]
        batch_labels = self.labels[batch_indices]
        
        # Pad/truncate to max_length
        if self.max_length:
            batch_lcs, mask = self._pad_sequences(batch_lcs, self.max_length)
        else:
            # Find max length in batch
            max_len = max(len(lc) for lc in batch_lcs)
            batch_lcs, mask = self._pad_sequences(batch_lcs, max_len)
        
        # Augment
        if self.augment:
            batch_lcs = self._augment_lightcurves(batch_lcs, mask)
        
        self.current_batch += 1
        return batch_lcs, batch_labels, mask
    
    def _pad_sequences(
        self,
        sequences: List[np.ndarray],
        max_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad or truncate sequences to uniform length.
        
        Args:
            sequences: List of variable-length arrays
            max_length: Target length
        
        Returns:
            Tuple of (padded_sequences, mask) where mask is 1 for valid, 0 for padding
        """
        batch_size = len(sequences)
        padded = np.zeros((batch_size, max_length), dtype=np.float32)
        mask = np.zeros((batch_size, max_length), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = seq[:length]
            mask[i, :length] = 1.0
        
        return padded, mask
    
    def _augment_lightcurves(
        self,
        lightcurves: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Apply data augmentation to light curves.
        
        Args:
            lightcurves: Batch of light curves (batch_size, max_timesteps)
            mask: Valid timestep mask
        
        Returns:
            Augmented light curves
        """
        # Add Gaussian noise to valid timesteps
        noise_level = 0.01
        noise = np.random.randn(*lightcurves.shape) * noise_level
        lightcurves = lightcurves + noise * mask
        
        # Random amplitude scaling
        scale = np.random.uniform(0.9, 1.1, (lightcurves.shape[0], 1))
        lightcurves = lightcurves * scale
        
        # Random time shift (circular shift for periodic signals)
        for i in range(lightcurves.shape[0]):
            valid_length = int(mask[i].sum())
            if valid_length > 0:
                shift = np.random.randint(0, valid_length)
                lightcurves[i, :valid_length] = np.roll(
                    lightcurves[i, :valid_length], shift
                )
        
        return lightcurves


def create_tensorflow_dataset(
    data_loader,
    output_types: Tuple = (np.float32, np.int32)
):
    """Create TensorFlow Dataset from data loader.
    
    Args:
        data_loader: SpectralDataLoader or LightCurveDataLoader
        output_types: Tuple of output data types
    
    Returns:
        tf.data.Dataset
    """
    try:
        import tensorflow as tf
        
        def generator():
            for batch in data_loader:
                if len(batch) == 2:  # Spectral data
                    yield batch
                else:  # Light curve data (includes mask)
                    yield batch[:2]  # Only return data and labels for now
        
        if isinstance(data_loader, SpectralDataLoader):
            output_signature = (
                tf.TensorSpec(shape=(None, data_loader.spectra.shape[1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        else:  # LightCurveDataLoader
            max_len = data_loader.max_length or 1000
            output_signature = (
                tf.TensorSpec(shape=(None, max_len), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    except ImportError:
        logger.warning("TensorFlow not installed; cannot create tf.data.Dataset")
        return None


def create_pytorch_dataloader(
    data_loader,
    num_workers: int = 0
):
    """Create PyTorch DataLoader from data loader.
    
    Args:
        data_loader: SpectralDataLoader or LightCurveDataLoader
        num_workers: Number of worker processes
    
    Returns:
        torch.utils.data.DataLoader
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        
        class WrapperDataset(Dataset):
            def __init__(self, loader):
                self.loader = loader
                self.data = loader.spectra if hasattr(loader, 'spectra') else loader.lightcurves
                self.labels = loader.labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                if isinstance(self.data, np.ndarray):
                    return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])
                else:
                    return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])
        
        dataset = WrapperDataset(data_loader)
        return DataLoader(
            dataset,
            batch_size=data_loader.batch_size,
            shuffle=data_loader.shuffle,
            num_workers=num_workers
        )
    
    except ImportError:
        logger.warning("PyTorch not installed; cannot create DataLoader")
        return None


__all__ = [
    'SpectralDataLoader',
    'LightCurveDataLoader',
    'create_tensorflow_dataset',
    'create_pytorch_dataloader'
]
