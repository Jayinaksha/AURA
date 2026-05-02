"""
Universal Gesture Recognition Model
Handles ANY gesture by reconstructing the actual trajectory path.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path

class SpatialAttention(layers.Layer):
    """Attention mechanism for spatial sensor data fusion."""
    
    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense1 = layers.Dense(units, activation='tanh')
        self.dense2 = layers.Dense(1, activation='softmax')
        
    def call(self, inputs):
        # inputs shape: (batch, time, features)
        attention_weights = self.dense2(self.dense1(inputs))
        weighted = inputs * attention_weights
        return weighted, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

class TrajectoryExtractor(layers.Layer):
    """Extracts 2D trajectory coordinates from multi-channel sensor data."""
    
    def __init__(self, coordinate_dim: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.coordinate_dim = coordinate_dim
        
        # Spatial processing layers
        self.spatial_conv1 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.spatial_conv2 = layers.Conv1D(32, 3, activation='relu', padding='same')
        self.spatial_attention = SpatialAttention(32)
        
        # Temporal processing
        self.temporal_lstm = layers.LSTM(64, return_sequences=True)
        self.temporal_dropout = layers.Dropout(0.2)
        
        # Coordinate prediction
        self.coord_dense1 = layers.Dense(32, activation='relu')
        self.coord_dense2 = layers.Dense(coordinate_dim, activation='tanh')  # Normalized coordinates
        
    def call(self, inputs, training=None):
        # Spatial feature extraction
        x = self.spatial_conv1(inputs)
        x = self.spatial_conv2(x)
        
        # Spatial attention
        x, attention_weights = self.spatial_attention(x)
        
        # Temporal modeling
        x = self.temporal_lstm(x)
        if training:
            x = self.temporal_dropout(x)
        
        # Coordinate prediction
        x = self.coord_dense1(x)
        coordinates = self.coord_dense2(x)  # Shape: (batch, time, 2)
        
        return coordinates, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'coordinate_dim': self.coordinate_dim})
        return config

class GestureClassifier(layers.Layer):
    """Optional classifier for gesture categories (parallel to trajectory)."""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # Feature extraction
        self.conv1 = layers.Conv1D(128, 5, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(2)
        self.conv2 = layers.Conv1D(64, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling1D(2)
        
        # Temporal modeling
        self.lstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))
        self.lstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=False))
        
        # Classification
        self.dropout = layers.Dropout(0.4)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.lstm1(x)
        x = self.lstm2(x)
        
        if training:
            x = self.dropout(x)
        
        x = self.dense1(x)
        classification = self.dense2(x)
        
        return classification
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

class UniversalGestureModel:
    """
    Universal Gesture Recognition Model that can handle ANY gesture.
    
    Primary Output: Trajectory coordinates (continuous path)
    Secondary Output: Gesture classification (optional, for known gestures)
    """
    
    def __init__(self,
                 sequence_length: int = 500,
                 n_features: int = 48,
                 num_classes: int = 17,
                 coordinate_dim: int = 2,
                 learning_rate: float = 0.001,
                 trajectory_weight: float = 0.8,
                 classification_weight: float = 0.2):
        """
        Initialize Universal Gesture Model.
        
        Args:
            sequence_length: Input sequence length
            n_features: Number of input features per timestep
            num_classes: Number of gesture classes (for optional classification)
            coordinate_dim: Coordinate dimensions (2 for x,y)
            learning_rate: Learning rate for training
            trajectory_weight: Weight for trajectory loss (primary)
            classification_weight: Weight for classification loss (secondary)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.num_classes = num_classes
        self.coordinate_dim = coordinate_dim
        self.learning_rate = learning_rate
        self.trajectory_weight = trajectory_weight
        self.classification_weight = classification_weight
        
        self.model = None
        self.trajectory_extractor = None
        self.gesture_classifier = None
        
    def build(self) -> keras.Model:
        """Build the universal gesture model."""
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features), name='sensor_input')
        
        # Trajectory extraction (PRIMARY OUTPUT)
        self.trajectory_extractor = TrajectoryExtractor(self.coordinate_dim)
        trajectories, spatial_attention = self.trajectory_extractor(inputs)
        
        # Gesture classification (SECONDARY OUTPUT)
        self.gesture_classifier = GestureClassifier(self.num_classes)
        classifications = self.gesture_classifier(inputs)
        
        # Create model with multiple outputs
        self.model = keras.Model(
            inputs=inputs,
            outputs={
                'trajectory': trajectories,  # Shape: (batch, time, 2) - PRIMARY
                'classification': classifications,  # Shape: (batch, classes) - SECONDARY  
                'attention': spatial_attention  # Shape: (batch, time, 1) - DEBUG
            },
            name='universal_gesture_model'
        )
        
        # Multi-task loss compilation
        self.model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                clipnorm=1.0
            ),
            loss={
                'trajectory': 'mse',  # Continuous coordinates
                'classification': 'sparse_categorical_crossentropy',  # Discrete classes
                'attention': None  # No loss for attention (debug only)
            },
            loss_weights={
                'trajectory': self.trajectory_weight,  # PRIMARY (80%)
                'classification': self.classification_weight,  # SECONDARY (20%)
                'attention': 0.0
            },
            metrics={
                'trajectory': ['mae', 'mse'],
                'classification': ['accuracy'],
                'attention': []
            }
        )
        
        return self.model
    
    def extract_coordinates_from_sensors(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Extract 2D coordinates from 5-plate sensor configuration.
        
        Args:
            sensor_data: Shape (batch, time, 12) - raw sensor channels
            
        Returns:
            coordinates: Shape (batch, time, 2) - normalized x,y coordinates
        """
        # 5-plate layout:
        # Channel mapping: [0,1,2,3,4] = [Left, Upper, Base, Right, Lower]
        
        batch_size, time_steps, _ = sensor_data.shape
        coordinates = np.zeros((batch_size, time_steps, 2))
        
        for b in range(batch_size):
            for t in range(time_steps):
                # Extract filtered values for each plate
                left = sensor_data[b, t, 0]    # Channel_0_Filtered
                upper = sensor_data[b, t, 2]   # Channel_1_Filtered  
                base = sensor_data[b, t, 4]    # Channel_2_Filtered (reference)
                right = sensor_data[b, t, 6]   # Channel_3_Filtered
                lower = sensor_data[b, t, 8]   # Channel_4_Filtered
                
                # Calculate relative position using plate differences
                # X-coordinate: Left vs Right plate activation
                x_raw = (right - left) / (abs(right) + abs(left) + 1e-6)
                
                # Y-coordinate: Upper vs Lower plate activation  
                y_raw = (upper - lower) / (abs(upper) + abs(lower) + 1e-6)
                
                # Normalize to [-1, 1] range
                coordinates[b, t, 0] = np.tanh(x_raw)  # X coordinate
                coordinates[b, t, 1] = np.tanh(y_raw)  # Y coordinate
                
        return coordinates
    
    def predict_trajectory(self, sensor_data: np.ndarray) -> Dict:
        """
        Predict trajectory and classification for sensor data.
        
        Args:
            sensor_data: Shape (batch, time, features) or (time, features)
            
        Returns:
            Dictionary with trajectory, classification, and confidence
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
            
        # Ensure batch dimension
        if len(sensor_data.shape) == 2:
            sensor_data = np.expand_dims(sensor_data, axis=0)
            
        # Make predictions
        predictions = self.model.predict(sensor_data, verbose=0)
        
        # Extract results
        trajectories = predictions['trajectory']  # (batch, time, 2)
        classifications = predictions['classification']  # (batch, classes)
        attention_weights = predictions['attention']  # (batch, time, 1)
        
        # Process results
        results = {
            'trajectory_path': trajectories[0],  # Remove batch dim: (time, 2)
            'predicted_class': np.argmax(classifications[0]),
            'class_confidence': np.max(classifications[0]),
            'class_probabilities': classifications[0],
            'attention_weights': attention_weights[0].flatten(),
            'trajectory_confidence': self._calculate_trajectory_confidence(trajectories[0])
        }
        
        return results
    
    def _calculate_trajectory_confidence(self, trajectory: np.ndarray) -> float:
        """Calculate confidence score for trajectory quality."""
        # Smooth trajectory = higher confidence
        velocity = np.diff(trajectory, axis=0)
        acceleration = np.diff(velocity, axis=0)
        
        # Lower acceleration variance = smoother trajectory = higher confidence
        smoothness = 1.0 / (1.0 + np.var(acceleration))
        
        # Path length consistency
        step_lengths = np.linalg.norm(velocity, axis=1)
        length_consistency = 1.0 / (1.0 + np.var(step_lengths))
        
        # Combined confidence
        confidence = 0.7 * smoothness + 0.3 * length_consistency
        return float(np.clip(confidence, 0.0, 1.0))
    
    def save_model(self, filepath: str, save_metadata: bool = True):
        """Save the complete model and metadata."""
        if self.model is None:
            raise ValueError("No model to save. Build and train first.")
            
        # Save model
        self.model.save(filepath)
        
        if save_metadata:
            # Save metadata
            metadata = {
                'model_type': 'universal_gesture',
                'version': '1.0',
                'architecture': {
                    'sequence_length': self.sequence_length,
                    'n_features': self.n_features,
                    'num_classes': self.num_classes,
                    'coordinate_dim': self.coordinate_dim
                },
                'training_config': {
                    'learning_rate': self.learning_rate,
                    'trajectory_weight': self.trajectory_weight,
                    'classification_weight': self.classification_weight
                },
                'outputs': {
                    'trajectory': 'Primary output - continuous 2D coordinates',
                    'classification': 'Secondary output - gesture categories',
                    'attention': 'Debug output - spatial attention weights'
                },
                'usage': {
                    'real_time': 'For live drawing applications',
                    'batch_processing': 'For offline gesture analysis',
                    'unlimited_gestures': 'Can handle any gesture via trajectory'
                }
            }
            
            metadata_path = Path(filepath).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        print(f"✅ Universal gesture model saved to: {filepath}")
        if save_metadata:
            print(f"✅ Metadata saved to: {metadata_path}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'UniversalGestureModel':
        """Load a saved universal gesture model."""
        # Load metadata if available
        metadata_path = Path(filepath).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                arch = metadata['architecture']
                config = metadata['training_config']
        else:
            # Default values if no metadata
            arch = {'sequence_length': 500, 'n_features': 48, 'num_classes': 17, 'coordinate_dim': 2}
            config = {'learning_rate': 0.001, 'trajectory_weight': 0.8, 'classification_weight': 0.2}
        
        # Create instance
        instance = cls(**arch, **config)
        
        # Load model with custom objects
        custom_objects = {
            'SpatialAttention': SpatialAttention,
            'TrajectoryExtractor': TrajectoryExtractor, 
            'GestureClassifier': GestureClassifier
        }
        
        instance.model = keras.models.load_model(filepath, custom_objects=custom_objects)
        
        print(f"✅ Universal gesture model loaded from: {filepath}")
        return instance
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build()
        
        print("=" * 60)
        print("🚀 UNIVERSAL GESTURE MODEL ARCHITECTURE")
        print("=" * 60)
        print(f"Input Shape: (batch, {self.sequence_length}, {self.n_features})")
        print(f"Primary Output: trajectory (batch, {self.sequence_length}, {self.coordinate_dim})")
        print(f"Secondary Output: classification (batch, {self.num_classes})")
        print(f"Loss Weights: Trajectory {self.trajectory_weight}, Classification {self.classification_weight}")
        print("=" * 60)
        
        self.model.summary()
