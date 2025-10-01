"""
Light curve variability models for stellar analysis.

This module provides deep learning models for classifying stellar variability
and estimating physical parameters from light curves.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
# Defer TensorFlow/Keras imports to runtime to make module lightweight for tests
tf = None  # type: ignore
keras = None  # type: ignore
layers = None  # type: ignore
import os
import json
from astropy.timeseries import TimeSeries  # type: ignore
from scipy.interpolate import interp1d  # Added for gap filling logic


class LightCurveProcessor:
    """
    Preprocessor for light curve data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the light curve processor.

        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or {}

    def preprocess(self, light_curve: Union[TimeSeries, np.ndarray]) -> np.ndarray:
        """
        Preprocess a light curve for model input.

        Args:
            light_curve: Input light curve as TimeSeries or array

        Returns:
            Preprocessed light curve array
        """
        # Convert to numpy array if needed
        if isinstance(light_curve, TimeSeries):
            flux = light_curve["flux"].value
            time = light_curve["time"].value
        else:
            flux = light_curve
            time = np.arange(len(flux))

        # Apply normalization
        flux = self._normalize_flux(flux)

        # Apply detrending if specified
        if self.config.get("detrend", False):
            flux = self._detrend_flux(flux, time)

        # Apply gap filling if specified
        if self.config.get("fill_gaps", False):
            flux = self._fill_gaps(flux, time)

        # Apply segmentation if specified
        if self.config.get("segment", False):
            segments = self._segment_lightcurve(flux, time)
            return segments

        return flux

    def _normalize_flux(self, flux: np.ndarray) -> np.ndarray:
        """
        Normalize flux values.

        Args:
            flux: Input flux array

        Returns:
            Normalized flux array
        """
        method = self.config.get("normalization", "standard")

        if method == "standard":
            return (flux - np.mean(flux)) / np.std(flux)
        elif method == "minmax":
            return (flux - np.min(flux)) / (np.max(flux) - np.min(flux))
        elif method == "robust":
            median = np.median(flux)
            mad = np.median(np.abs(flux - median))
            return (flux - median) / mad
        else:
            return flux

    def _detrend_flux(self, flux: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        Remove trends from flux values.

        Args:
            flux: Input flux array
            time: Time array

        Returns:
            Detrended flux array
        """
        method = self.config.get("detrend_method", "polynomial")

        if method == "polynomial":
            degree = self.config.get("detrend_degree", 2)
            coeffs = np.polyfit(time, flux, degree)
            trend = np.polyval(coeffs, time)
            return flux - trend
        elif method == "spline":
            from scipy.interpolate import UnivariateSpline
            smoothing = self.config.get("detrend_smoothing", 10)
            spline = UnivariateSpline(time, flux, s=smoothing)
            trend = spline(time)
            return flux - trend
        else:
            return flux

    def _fill_gaps(self, flux: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        Fill gaps in light curve data.

        Args:
            flux: Input flux array
            time: Time array

        Returns:
            Gap-filled flux array
        """
        method = self.config.get("fill_method", "linear")

        if method == "linear":
            # Identify large gaps by time spacing
            time_diffs = np.diff(time)
            median_dt = np.median(time_diffs) if len(time_diffs) > 0 else 1.0
            gap_threshold = self.config.get("gap_threshold", median_dt * 5)
            gap_indices = np.where(time_diffs > gap_threshold)[0]

            filled = flux.copy()
            for idx in gap_indices:
                # indices between idx and idx+1 are missing; we interpolate a ramp
                t1, t2 = time[idx], time[idx + 1]
                f1, f2 = filled[idx], filled[idx + 1]
                if not np.isfinite(f1) or not np.isfinite(f2):
                    # if edges are nan, skip
                    continue
                span = int(round((t2 - t1) / median_dt))
                if span <= 1:
                    continue
                # Linear values from f1 to f2 with span+1 samples; overwrite interior
                ramp = np.linspace(f1, f2, span + 1)
                # place interior points into filled (no actual intermediate indices exist; best-effort)
                # choose closest index range
                # Here we simply smooth neighbors to reduce discontinuity
                filled[idx] = (filled[idx] + ramp[0]) / 2.0
                filled[idx + 1] = (filled[idx + 1] + ramp[-1]) / 2.0
            # If there are NaNs (from original missing points), interpolate over them directly
            if np.any(~np.isfinite(filled)):
                n = len(filled)
                good = np.isfinite(filled)
                if good.any():
                    filled[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), filled[good])
            return filled
        else:
            return flux

    def _segment_lightcurve(self, flux: np.ndarray, time: np.ndarray) -> List[np.ndarray]:
        """
        Segment a light curve into chunks.

        Args:
            flux: Input flux array
            time: Time array

        Returns:
            List of flux segments
        """
        segment_length = self.config.get("segment_length", 100)
        overlap = self.config.get("segment_overlap", 0.1)

        segments = []
        step = int(segment_length * (1 - overlap))

        for i in range(0, len(flux) - segment_length + 1, step):
            segments.append(flux[i:i+segment_length])

        return segments


class LightCurveClassifier:
    """
    Base class for light curve classification models.
    """

    def __init__(self, input_shape: Tuple[int, int], num_classes: int, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the light curve classifier.

        Args:
            input_shape: Shape of input light curves (time points, features)
            num_classes: Number of variability classes
            config: Configuration dictionary for model parameters
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config or {}
        self.model = None
        self.history = None
        self.processor = LightCurveProcessor(self.config)

    def build_model(self):  # returns tf.keras.Model
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

        if self.model is None:
            raise ValueError("Model has not been built")
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
            x_train: Training light curves
            y_train: Training labels
            x_val: Validation light curves
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
        if keras is not None:
            if not any(isinstance(cb, getattr(keras.callbacks, 'EarlyStopping')) for cb in callbacks):
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
            x: Input light curves

        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        # Preprocess input
        if len(x.shape) == 2:  # Single light curve
            x = np.expand_dims(x, axis=0)

        # Ensure processor has preprocess method
        preprocess_fn = getattr(self.processor, 'preprocess', None)
        if preprocess_fn is None:
            raise AttributeError('LightCurveProcessor is missing preprocess method')
        processed_x = np.array([preprocess_fn(lc) for lc in x])

        return self.model.predict(processed_x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            x: Test light curves
            y: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        # Preprocess input
        if len(x.shape) == 2:  # Single light curve
            x = np.expand_dims(x, axis=0)

        preprocess_fn = getattr(self.processor, 'preprocess', None)
        if preprocess_fn is None:
            raise AttributeError('LightCurveProcessor is missing preprocess method')
        processed_x = np.array([preprocess_fn(lc) for lc in x])

        results = self.model.evaluate(processed_x, y, return_dict=True)
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


class LightCurveTransformer(LightCurveClassifier):
    """
    Transformer-based light curve classifier for stellar variability.
    """

    def build_model(self):  # returns tf.keras.Model
        """
        Build the Transformer model architecture.

        Returns:
            Compiled Keras model
        """
        global tf, keras, layers
        if keras is None or layers is None:
            try:
                import tensorflow as _tf  # type: ignore
                tf = _tf
                keras = _tf.keras
                layers = keras.layers
            except Exception as e:  # pragma: no cover
                raise RuntimeError("TensorFlow/Keras is required to build LightCurveTransformer but is not available.") from e
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

    def _transformer_block(self, x: Any, num_heads: int, ff_dim: int) -> Any:
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


class LightCurveMultiTask(LightCurveClassifier):
    """
    Multi-task light curve classifier that predicts both variability classes and physical parameters.
    """

    def __init__(self, input_shape: Tuple[int, int], num_classes: int, 
                 num_parameters: int, parameter_names: List[str],
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-task light curve classifier.

        Args:
            input_shape: Shape of input light curves (time points, features)
            num_classes: Number of variability classes
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
        # Use a base architecture (e.g., Transformer)
        base_model = LightCurveTransformer(self.input_shape, self.num_classes, self.config)
        base_model.build_model()

        # Remove the original output layer
        base_model.model.pop()

        # Get the output of the second-to-last layer
        x = base_model.model.layers[-1].output

        # Add dropout for regularization
        x = layers.Dropout(0.5)(x)

        # Create two output branches
        # Classification branch for variability classes
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
            x_train: Training light curves
            y_train: Dictionary with training labels and parameters
            x_val: Validation light curves
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
            x: Input light curves

        Returns:
            Dictionary with predicted class probabilities and parameters
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        # Preprocess input
        if len(x.shape) == 2:  # Single light curve
            x = np.expand_dims(x, axis=0)

        preprocess_fn = getattr(self.processor, 'preprocess', None)
        if preprocess_fn is None:
            raise AttributeError('LightCurveProcessor is missing preprocess method')
        processed_x = np.array([preprocess_fn(lc) for lc in x])

        predictions = self.model.predict(processed_x)
        return {
            "class_output": predictions[0],
            "param_output": predictions[1]
        }

    def evaluate(self, x: np.ndarray, y: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            x: Test light curves
            y: Dictionary with test labels and parameters

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained")

        # Preprocess input
        if len(x.shape) == 2:  # Single light curve
            x = np.expand_dims(x, axis=0)

        # Use preprocess (previously incorrect attribute 'process')
        preprocess_fn = getattr(self.processor, 'preprocess', None)
        if preprocess_fn is None:
            raise AttributeError('LightCurveProcessor is missing preprocess method')
        processed_x = np.array([preprocess_fn(lc) for lc in x])

        results = self.model.evaluate(processed_x, y, return_dict=True)
        return results
