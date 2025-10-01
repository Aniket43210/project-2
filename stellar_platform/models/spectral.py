"""
Spectral classification models for stellar analysis.

This module provides deep learning models for classifying stellar spectra,
including 1D CNN and Transformer architectures.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Any, Union

# Use unified tf.keras (standalone keras previously caused env drift)
keras = tf.keras  # type: ignore
layers = keras.layers  # shorthand
import os
import json


class SpectralClassifier:
    """
    Base class for spectral classification models.
    """

    def __init__(self, input_shape: Tuple[int, int], num_classes: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the spectral classifier.

        Args:
            input_shape: Shape of input spectra (wavelengths, flux)
            num_classes: Number of stellar classes
            config: Configuration dictionary for model parameters
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config or {}
        self.model = None
        self.history = None

    def build_model(self):  # -> keras.Model (runtime fallback safe)
        """
        Build the model architecture.

        Returns:
            Compiled Keras model
        """
        raise NotImplementedError("Subclasses must implement build_model")

    def compile_model(self, optimizer: str = "adam", loss: str = "categorical_crossentropy", metrics: List[str] = None):
        """
        Compile the model.

        Args:
            optimizer: Optimization algorithm
            loss: Loss function
            metrics: List of metrics to track
        """
        if metrics is None:
            metrics = ["accuracy"]

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
    callbacks: Optional[List[Any]] = None,
        class_weight: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            x_train: Training spectra
            y_train: Training labels
            x_val: Validation spectra
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            callbacks: List of callbacks to use during training
            class_weight: Class weights for handling imbalance

        Returns:
            Dictionary with training history
        """
        if self.model is None:
            self.build_model()

        if callbacks is None:
            callbacks = []

        # Add early stopping callback if not provided
        if not any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks):
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        # Train the model
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight
        )

        return self.history.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            x: Input spectra

        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        return self.model.predict(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            x: Test spectra
            y: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        results = self.model.evaluate(x, y, return_dict=True)
        return results

    def save_model(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Normalize extension to native Keras format
        from pathlib import Path as _Path
        p = _Path(path)
        if p.suffix.lower() not in {'.keras'}:
            p = p.with_suffix('.keras')
        # Save model and config
        self.model.save(str(p))
        config_path = str(p.with_suffix('')) + "_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "config": self.config
            }, f)

    def load_model(self, path: str):
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        # Load model (supports .keras)
        from pathlib import Path as _Path
        p = _Path(path)
        if p.suffix.lower() not in {'.keras'}:
            # try native format by switching suffix
            cand = p.with_suffix('.keras')
            if cand.exists():
                p = cand
        self.model = keras.models.load_model(str(p))

        # Load config
        config_path = str(p.with_suffix('')) + "_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.input_shape = tuple(config["input_shape"])
                self.num_classes = config["num_classes"]
                self.config = config["config"]


class SpectralCNN(SpectralClassifier):
    """
    1D CNN-based spectral classifier.
    """

    def build_model(self):  # returns tf.keras.Model
        """
        Build the 1D CNN model architecture.

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = keras.Input(shape=self.input_shape)

        # Initial convolutional block
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv1D(filters=32, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # Additional convolutional blocks
        x = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(filters=128, kernel_size=3, activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)

        self.model = model
        return model


class SpectralTransformer(SpectralClassifier):
    """
    Transformer-based spectral classifier.
    """

    def build_model(self):  # returns tf.keras.Model
        """
        Build the Transformer model architecture.

        Returns:
            Compiled Keras model
        """
        # Get model parameters from config
        num_heads = self.config.get("num_heads", 4)
        ff_dim = self.config.get("ff_dim", 128)
        num_transformer_blocks = self.config.get("num_transformer_blocks", 2)
        mlp_units = self.config.get("mlp_units", [128, 64])

        # Input layer
        inputs = keras.Input(shape=self.input_shape)

        # Initial normalization
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Normalization()(x)

        # Add positional encoding
        positions = tf.range(start=0, limit=self.input_shape[0], delta=1)
        position_embedding = layers.Embedding(
            input_dim=self.input_shape[0],
            output_dim=self.input_shape[1]
        )(positions)
        x = x + position_embedding

        # Transformer blocks
        for _ in range(num_transformer_blocks):
            x = self._transformer_block(x, num_heads, ff_dim)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        # MLP head
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(0.1)(x)

        # Output layer
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)

        self.model = model
        return model

    def _transformer_block(self, x: tf.Tensor, num_heads: int, ff_dim: int) -> tf.Tensor:
        """
        Create a transformer block.

        Args:
            x: Input tensor
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension

        Returns:
            Output tensor
        """
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=self.input_shape[1]
        )(x, x)
        attention_output = layers.Dropout(0.1)(attention_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Feed-forward network
        ffn_output = layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = layers.Dense(self.input_shape[1])(ffn_output)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

        return x


class SpectralMultiTask(SpectralClassifier):
    """
    Multi-task spectral classifier that predicts both stellar classes and parameters.
    """

    def __init__(self, input_shape: Tuple[int, int], num_classes: int, 
                 num_parameters: int, parameter_names: List[str],
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-task spectral classifier.

        Args:
            input_shape: Shape of input spectra (wavelengths, flux)
            num_classes: Number of stellar classes
            num_parameters: Number of physical parameters to predict
            parameter_names: Names of the physical parameters
            config: Configuration dictionary for model parameters
        """
        super().__init__(input_shape, num_classes, config)
        self.num_parameters = num_parameters
        self.parameter_names = parameter_names

    def build_model(self):  # returns tf.keras.Model
        """
        Build the multi-task model architecture.

        Returns:
            Compiled Keras model
        """
        # Use a base architecture (e.g., CNN or Transformer)
        base_architecture = self.config.get("base_architecture", "cnn")

        if base_architecture == "cnn":
            base_model = SpectralCNN(self.input_shape, self.num_classes, self.config)
        elif base_architecture == "transformer":
            base_model = SpectralTransformer(self.input_shape, self.num_classes, self.config)
        else:
            raise ValueError(f"Unknown base architecture: {base_architecture}")

        # Build the base model
        base_model.build_model()

        # Remove the original output layer
        base_model.model.pop()

        # Get the output of the second-to-last layer
        x = base_model.model.layers[-1].output

        # Add dropout for regularization
        x = layers.Dropout(0.5)(x)

        # Create two output branches
        # Classification branch
        class_output = layers.Dense(self.num_classes, activation="softmax", name="class_output")(x)

        # Regression branch for physical parameters
        param_output = layers.Dense(self.num_parameters, activation="linear", name="param_output")(x)

        # Create model with multiple outputs
        model = keras.Model(inputs=base_model.model.inputs, 
                           outputs=[class_output, param_output])

        self.model = model
        return model

    def compile_model(self, optimizer: str = "adam", 
                     class_loss: str = "categorical_crossentropy",
                     param_loss: str = "mse",
                     metrics: List[str] = None,
                     class_weight: Optional[Dict[int, float]] = None):
        """
        Compile the multi-task model.

        Args:
            optimizer: Optimization algorithm
            class_loss: Loss function for classification
            param_loss: Loss function for regression
            metrics: List of metrics to track
            class_weight: Class weights for handling imbalance
        """
        if metrics is None:
            metrics = ["accuracy"]

        self.model.compile(
            optimizer=optimizer,
            loss={
                "class_output": class_loss,
                "param_output": param_loss
            },
            metrics={
                "class_output": metrics,
                "param_output": ["mae"]
            },
            loss_weights={
                "class_output": 1.0,
                "param_output": 0.5  # Adjust based on importance
            }
        )

    def train(
        self,
        x_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        x_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        batch_size: int = 32,
        epochs: int = 100,
    callbacks: Optional[List[Any]] = None,
        class_weight: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Train the multi-task model.

        Args:
            x_train: Training spectra
            y_train: Dictionary with training labels and parameters
            x_val: Validation spectra
            y_val: Dictionary with validation labels and parameters
            batch_size: Batch size for training
            epochs: Number of training epochs
            callbacks: List of callbacks to use during training
            class_weight: Class weights for handling imbalance

        Returns:
            Dictionary with training history
        """
        if self.model is None:
            self.build_model()

        if callbacks is None:
            callbacks = []

        # Add early stopping callback if not provided
        if not any(isinstance(cb, keras.callbacks.EarlyStopping) for cb in callbacks):
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        # Train the model
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight
        )

        return self.history.history

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            x: Input spectra

        Returns:
            Dictionary with predicted class probabilities and parameters
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        predictions = self.model.predict(x)
        return {
            "class_output": predictions[0],
            "param_output": predictions[1]
        }

    def evaluate(self, x: np.ndarray, y: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            x: Test spectra
            y: Dictionary with test labels and parameters

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        results = self.model.evaluate(x, y, return_dict=True)
        return results
