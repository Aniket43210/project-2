"""
SED (Spectral Energy Distribution) models for stellar analysis.

This module provides gradient boosting models for classifying stellar types
and estimating physical parameters from photometric data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import json
import joblib
from pathlib import Path


class SEDProcessor:
    """
    Preprocessor for SED data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SED processor.

        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def fit(self, sed_data: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Fit the processor on training data.

        Args:
            sed_data: Training SED data
            labels: Training labels (optional)
        """
        # Store feature names if provided
        if self.feature_names is None and len(sed_data.shape) == 2:
            self.feature_names = [f"feature_{i}" for i in range(sed_data.shape[1])]

        # Fit scaler
        self.scaler.fit(sed_data)

        # Fit label encoder if labels are provided
        if labels is not None:
            self.label_encoder.fit(labels)

    def transform(self, sed_data: np.ndarray) -> np.ndarray:
        """
        Transform SED data using fitted processor.

        Args:
            sed_data: Input SED data

        Returns:
            Transformed SED data
        """
        # Scale features
        scaled_data = self.scaler.transform(sed_data)

        return scaled_data

    def fit_transform(self, sed_data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the processor and transform the data.

        Args:
            sed_data: Training SED data
            labels: Training labels (optional)

        Returns:
            Transformed SED data
        """
        self.fit(sed_data, labels)
        return self.transform(sed_data)

    def inverse_transform(self, transformed_data: np.ndarray) -> np.ndarray:
        """
        Inverse transform the scaled data.

        Args:
            transformed_data: Transformed SED data

        Returns:
            Original scale SED data
        """
        return self.scaler.inverse_transform(transformed_data)

    def encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Encode string labels to integers.

        Args:
            labels: Input labels

        Returns:
            Encoded labels
        """
        return self.label_encoder.transform(labels)

    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Decode integer labels back to strings.

        Args:
            encoded_labels: Encoded labels

        Returns:
            Decoded labels
        """
        return self.label_encoder.inverse_transform(encoded_labels)


class SEDClassifier:
    """
    Gradient boosting classifier for SED-based stellar classification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SED classifier.

        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or {}
        self.model = None
        self.processor = SEDProcessor(self.config)
        self.num_classes = None
        self.class_names = None

    def build_model(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Build the gradient boosting classifier.

        Args:
            num_classes: Number of stellar classes
            class_names: Names of the classes (optional)
        """
        # Get model parameters from config
        n_estimators = self.config.get("n_estimators", 100)
        learning_rate = self.config.get("learning_rate", 0.1)
        max_depth = self.config.get("max_depth", 3)
        min_samples_split = self.config.get("min_samples_split", 2)
        min_samples_leaf = self.config.get("min_samples_leaf", 1)

        # Create the classifier
        base_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        # Wrap for multi-class classification
        self.model = MultiOutputClassifier(base_model)
        self.num_classes = num_classes
        self.class_names = class_names

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weight: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Train the classifier.

        Args:
            x_train: Training SED data
            y_train: Training labels
            x_val: Validation SED data (optional)
            y_val: Validation labels (optional)
            class_weight: Class weights for handling imbalance

        Returns:
            Dictionary with training history
        """
        # Fit processor on training data
        x_train_processed = self.processor.fit_transform(x_train, y_train)

        # Encode labels
        y_train_encoded = self.processor.encode_labels(y_train)

        # Train the model
        self.model.fit(x_train_processed, y_train_encoded)

        # Evaluate on validation set if provided
        results = {"train_score": self.model.score(x_train_processed, y_train_encoded)}

        if x_val is not None and y_val is not None:
            x_val_processed = self.processor.transform(x_val)
            y_val_encoded = self.processor.encode_labels(y_val)
            results["val_score"] = self.model.score(x_val_processed, y_val_encoded)

        return results

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            x: Input SED data

        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        # Preprocess input
        x_processed = self.processor.transform(x)

        # Make predictions
        predictions = self.model.predict_proba(x_processed)

        # Aggregate predictions if multi-output
        if isinstance(predictions, list):
            # For multi-output, take the mean probability across outputs
            predictions = np.mean([p[:, 1] for p in predictions], axis=0)

        return predictions

    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """
        Make class predictions on new data.

        Args:
            x: Input SED data

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        # Preprocess input
        x_processed = self.processor.transform(x)

        # Make predictions
        predictions = self.model.predict(x_processed)

        # Decode labels
        if self.class_names is not None:
            predictions = self.processor.decode_labels(predictions)

        return predictions

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            x: Test SED data
            y: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        # Preprocess input
        x_processed = self.processor.transform(x)
        y_encoded = self.processor.encode_labels(y)

        # Evaluate
        score = self.model.score(x_processed, y_encoded)

        return {"accuracy": score}

    def save_model(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model and processor
        model_path = path.replace(".pkl", "_model.pkl")
        processor_path = path.replace(".pkl", "_processor.pkl")

        joblib.dump(self.model, model_path)
        joblib.dump(self.processor, processor_path)

        # Save config and metadata
        metadata = {
            "config": self.config,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "feature_names": self.processor.feature_names
        }

        metadata_path = path.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def load_model(self, path: str):
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        # Load model and processor
        model_path = path.replace(".pkl", "_model.pkl")
        processor_path = path.replace(".pkl", "_processor.pkl")

        self.model = joblib.load(model_path)
        self.processor = joblib.load(processor_path)

        # Load metadata
        metadata_path = path.replace(".pkl", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.config = metadata["config"]
                self.num_classes = metadata["num_classes"]
                self.class_names = metadata["class_names"]
                self.processor.feature_names = metadata["feature_names"]


class SEDRegressor:
    """
    Gradient boosting regressor for estimating stellar parameters from SED data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SED regressor.

        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or {}
        self.model = None
        self.processor = SEDProcessor(self.config)
        self.num_parameters = None
        self.parameter_names = None

    def build_model(self, num_parameters: int, parameter_names: Optional[List[str]] = None):
        """
        Build the gradient boosting regressor.

        Args:
            num_parameters: Number of physical parameters to predict
            parameter_names: Names of the parameters (optional)
        """
        # Get model parameters from config
        n_estimators = self.config.get("n_estimators", 100)
        learning_rate = self.config.get("learning_rate", 0.1)
        max_depth = self.config.get("max_depth", 3)
        min_samples_split = self.config.get("min_samples_split", 2)
        min_samples_leaf = self.config.get("min_samples_leaf", 1)

        # Create the regressor
        base_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        # Wrap for multi-output regression
        self.model = MultiOutputRegressor(base_model)
        self.num_parameters = num_parameters
        self.parameter_names = parameter_names

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the regressor.

        Args:
            x_train: Training SED data
            y_train: Training parameters
            x_val: Validation SED data (optional)
            y_val: Validation parameters (optional)

        Returns:
            Dictionary with training history
        """
        # Fit processor on training data
        x_train_processed = self.processor.fit_transform(x_train)

        # Train the model
        self.model.fit(x_train_processed, y_train)

        # Evaluate on validation set if provided
        results = {"train_score": self.model.score(x_train_processed, y_train)}

        if x_val is not None and y_val is not None:
            x_val_processed = self.processor.transform(x_val)
            results["val_score"] = self.model.score(x_val_processed, y_val)

        return results

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            x: Input SED data

        Returns:
            Predicted physical parameters
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        # Preprocess input
        x_processed = self.processor.transform(x)

        # Make predictions
        predictions = self.model.predict(x_processed)

        return predictions

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            x: Test SED data
            y: Test parameters

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        # Preprocess input
        x_processed = self.processor.transform(x)

        # Evaluate
        score = self.model.score(x_processed, y)

        # Calculate MAE for each parameter
        predictions = self.model.predict(x_processed)
        mae = np.mean(np.abs(predictions - y), axis=0)

        results = {"r2_score": score}
        for i, name in enumerate(self.parameter_names or [f"param_{i}" for i in range(y.shape[1])]):
            results[f"mae_{name}"] = mae[i]

        return results

    def save_model(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model and processor
        model_path = path.replace(".pkl", "_model.pkl")
        processor_path = path.replace(".pkl", "_processor.pkl")

        joblib.dump(self.model, model_path)
        joblib.dump(self.processor, processor_path)

        # Save config and metadata
        metadata = {
            "config": self.config,
            "num_parameters": self.num_parameters,
            "parameter_names": self.parameter_names,
            "feature_names": self.processor.feature_names
        }

        metadata_path = path.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def load_model(self, path: str):
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        # Load model and processor
        model_path = path.replace(".pkl", "_model.pkl")
        processor_path = path.replace(".pkl", "_processor.pkl")

        self.model = joblib.load(model_path)
        self.processor = joblib.load(processor_path)

        # Load metadata
        metadata_path = path.replace(".pkl", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.config = metadata["config"]
                self.num_parameters = metadata["num_parameters"]
                self.parameter_names = metadata["parameter_names"]
                self.processor.feature_names = metadata["feature_names"]


class SEDMultiTask:
    """
    Multi-task model that combines classification and regression for SED analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SED multi-task model.

        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or {}
        self.classifier = SEDClassifier(self.config)
        self.regressor = SEDRegressor(self.config)
        self.processor = SEDProcessor(self.config)
        self.num_classes = None
        self.num_parameters = None
        self.class_names = None
        self.parameter_names = None

    def build_model(self, num_classes: int, num_parameters: int,
                   class_names: Optional[List[str]] = None,
                   parameter_names: Optional[List[str]] = None):
        """
        Build the multi-task model.

        Args:
            num_classes: Number of stellar classes
            num_parameters: Number of physical parameters to predict
            class_names: Names of the classes (optional)
            parameter_names: Names of the parameters (optional)
        """
        self.num_classes = num_classes
        self.num_parameters = num_parameters
        self.class_names = class_names
        self.parameter_names = parameter_names

        # Build individual models
        self.classifier.build_model(num_classes, class_names)
        self.regressor.build_model(num_parameters, parameter_names)

    def train(
        self,
        x_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[Dict[str, np.ndarray]] = None,
        class_weight: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Train the multi-task model.

        Args:
            x_train: Training SED data
            y_train: Dictionary with training labels and parameters
            x_val: Validation SED data (optional)
            y_val: Dictionary with validation labels and parameters (optional)
            class_weight: Class weights for handling imbalance

        Returns:
            Dictionary with training history
        """
        # Fit processor on training data
        x_train_processed = self.processor.fit_transform(x_train, y_train["labels"])

        # Train classifier
        classifier_results = self.classifier.train(
            x_train_processed, y_train["labels"],
            x_val, y_val["labels"] if y_val else None,
            class_weight
        )

        # Train regressor
        regressor_results = self.regressor.train(
            x_train_processed, y_train["parameters"],
            x_val, y_val["parameters"] if y_val else None
        )

        # Combine results
        results = {
            "classifier": classifier_results,
            "regressor": regressor_results
        }

        return results

    def predict(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            x: Input SED data

        Returns:
            Dictionary with predicted class probabilities and parameters
        """
        # Preprocess input
        x_processed = self.processor.transform(x)

        # Make predictions
        class_probs = self.classifier.predict(x_processed)
        parameters = self.regressor.predict(x_processed)

        return {
            "class_probabilities": class_probs,
            "parameters": parameters
        }

    def predict_classes(self, x: np.ndarray) -> np.ndarray:
        """
        Make class predictions on new data.

        Args:
            x: Input SED data

        Returns:
            Predicted class labels
        """
        # Preprocess input
        x_processed = self.processor.transform(x)

        # Make predictions
        return self.classifier.predict_classes(x_processed)

    def evaluate(self, x: np.ndarray, y: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            x: Test SED data
            y: Dictionary with test labels and parameters

        Returns:
            Dictionary with evaluation metrics
        """
        # Preprocess input
        x_processed = self.processor.transform(x)

        # Evaluate classifier
        classifier_results = self.classifier.evaluate(x_processed, y["labels"])

        # Evaluate regressor
        regressor_results = self.regressor.evaluate(x_processed, y["parameters"])

        # Combine results
        results = {
            "classifier": classifier_results,
            "regressor": regressor_results
        }

        return results

    def save_model(self, path: str):
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if self.classifier.model is None or self.regressor.model is None:
            raise ValueError("Models have not been trained")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save models and processor
        classifier_path = path.replace(".pkl", "_classifier.pkl")
        regressor_path = path.replace(".pkl", "_regressor.pkl")
        processor_path = path.replace(".pkl", "_processor.pkl")

        self.classifier.save_model(classifier_path)
        self.regressor.save_model(regressor_path)
        joblib.dump(self.processor, processor_path)

        # Save config and metadata
        metadata = {
            "config": self.config,
            "num_classes": self.num_classes,
            "num_parameters": self.num_parameters,
            "class_names": self.class_names,
            "parameter_names": self.parameter_names,
            "feature_names": self.processor.feature_names
        }

        metadata_path = path.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def load_model(self, path: str):
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        # Load models and processor
        classifier_path = path.replace(".pkl", "_classifier.pkl")
        regressor_path = path.replace(".pkl", "_regressor.pkl")
        processor_path = path.replace(".pkl", "_processor.pkl")

        self.classifier.load_model(classifier_path)
        self.regressor.load_model(regressor_path)
        self.processor = joblib.load(processor_path)

        # Load metadata
        metadata_path = path.replace(".pkl", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self.config = metadata["config"]
                self.num_classes = metadata["num_classes"]
                self.num_parameters = metadata["num_parameters"]
                self.class_names = metadata["class_names"]
                self.parameter_names = metadata["parameter_names"]
                self.processor.feature_names = metadata["feature_names"]
