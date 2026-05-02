import serial
import csv
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Tuple, Optional
import time
import math
import datetime
import tkinter as tk
from collections import deque
import joblib
import pandas as pd
from collections import deque


class SpatialAttentionLayer(layers.Layer):
    """Attention mechanism optimized for trajectory prediction."""
    
    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W_attention = layers.Dense(units, activation='tanh')
        self.V_attention = layers.Dense(1, use_bias=False)
        
    def call(self, inputs):
        # inputs shape: (batch, time, features)
        # Calculate attention scores
        score = self.V_attention(self.W_attention(inputs))  # (batch, time, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Apply attention
        context = inputs * attention_weights
        return context, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class TrajectoryOnlyModel:
    """
    CNN-BiLSTM model specialized for trajectory reconstruction.
    
    Architecture: Multi-scale CNN → BiLSTM → Attention → Dense → Trajectory Coordinates
    
    Key Features:
    - Trajectory-only output (no classification overhead)
    - Multi-scale CNN for spatial feature extraction
    - Bidirectional LSTM for temporal modeling
    - Attention mechanism for focusing on key timesteps
    - CPU-optimized with memory efficiency
    """
    
    def __init__(self,
                 sequence_length: int = 500,
                 n_features: int = 48,
                 coordinate_dim: int = 2,
                 lstm_units: int = 32,  # Reduced from 64
                 cnn_filters: Tuple[int, ...] = (32, 16, 8),  # Reduced from (64, 32, 16)
                 dropout_rate: float = 0.5,  # Increased from 0.3
                 learning_rate: float = 0.001):
        """
        Initialize trajectory-only model.
        
        Args:
            sequence_length: Input sequence length (timesteps)
            n_features: Number of input features per timestep
            coordinate_dim: Output coordinate dimensions (2 for x,y)
            lstm_units: Number of LSTM units per layer
            cnn_filters: CNN filter sizes for multi-scale feature extraction
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.coordinate_dim = coordinate_dim
        self.lstm_units = lstm_units
        self.cnn_filters = cnn_filters
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self._build_architecture()
    
    def _build_architecture(self):
        """Build the trajectory-only model architecture."""
        # Input layer
        inputs = layers.Input(
            shape=(self.sequence_length, self.n_features),
            name='sensor_features'
        )
        
        # Multi-scale CNN feature extraction
        conv_outputs = []
        for i, filters in enumerate(self.cnn_filters):
            # Use different kernel sizes for multi-scale features
            kernel_size = 3 + (i * 2)  # 3, 5, 7
            
            conv = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(inputs)
            conv = layers.BatchNormalization(name=f'bn_{i+1}')(conv)
            conv_outputs.append(conv)
        
        # Concatenate multi-scale features
        if len(conv_outputs) > 1:
            x = layers.Concatenate(axis=-1, name='multi_scale_concat')(conv_outputs)
        else:
            x = conv_outputs[0]
        
        # Note: Removed MaxPooling1D to maintain sequence length for trajectory output
        # This ensures output matches input sequence length (500)
        
        # Bidirectional LSTM layers for temporal modeling
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True),
            name='bilstm_1'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_1')(x)
        
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units // 2, return_sequences=True),
            name='bilstm_2'
        )(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_2')(x)
        
        # Spatial attention for trajectory focus (reduced units)
        x, attention_weights = SpatialAttentionLayer(
            units=32, name='spatial_attention'  # Reduced from 64
        )(x)
        
        # Trajectory coordinate prediction (reduced units)
        x = layers.TimeDistributed(
            layers.Dense(16, activation='relu'),  # Reduced from 32
            name='trajectory_dense_1'
        )(x)
        x = layers.TimeDistributed(
            layers.Dropout(self.dropout_rate),
            name='trajectory_dropout'
        )(x)
        
        # Output layer: (x, y) coordinates normalized to [-1, 1]
        trajectory_coords = layers.TimeDistributed(
            layers.Dense(self.coordinate_dim, activation='tanh'),
            name='trajectory_coordinates'
        )(x)
        
        # Create model
        self.model = keras.Model(
            inputs=inputs,
            outputs=trajectory_coords,
            name='trajectory_only_model'
        )
        
        # Custom loss functions for trajectory reconstruction
        def correlation_loss(y_true, y_pred):
            """Loss that maximizes correlation between predicted and true trajectories."""
            # Calculate correlation per coordinate dimension (x and y separately)
            # Shape: (batch, time, 2)
            
            # For each coordinate dimension
            correlations = []
            for dim in range(2):
                true_dim = tf.reshape(y_true[:, :, dim], [-1])  # Flatten
                pred_dim = tf.reshape(y_pred[:, :, dim], [-1])  # Flatten
                
                # Normalize
                true_mean = tf.reduce_mean(true_dim)
                pred_mean = tf.reduce_mean(pred_dim)
                true_norm = true_dim - true_mean
                pred_norm = pred_dim - pred_mean
                
                # Calculate correlation
                numerator = tf.reduce_sum(true_norm * pred_norm)
                true_std = tf.sqrt(tf.reduce_sum(tf.square(true_norm)) + 1e-8)
                pred_std = tf.sqrt(tf.reduce_sum(tf.square(pred_norm)) + 1e-8)
                denominator = true_std * pred_std + 1e-8
                
                correlation = numerator / denominator
                correlations.append(correlation)
            
            # Average correlation across dimensions, convert to loss
            mean_correlation = tf.reduce_mean(correlations)
            correlation_loss_val = 1.0 - mean_correlation
            
            return correlation_loss_val
        
        def velocity_loss(y_true, y_pred):
            """Loss on trajectory velocity (direction and speed)."""
            # Calculate velocities (differences between consecutive points)
            true_vel = y_true[:, 1:, :] - y_true[:, :-1, :]
            pred_vel = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            
            # MSE on velocities
            vel_mse = tf.reduce_mean(tf.square(true_vel - pred_vel))
            return vel_mse
        
        def trajectory_combined_loss(y_true, y_pred):
            """Combined loss: correlation + velocity + MSE."""
            # MSE loss (Keras 3.x compatible)
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            
            # Correlation loss
            corr_loss = correlation_loss(y_true, y_pred)
            
            # Velocity loss
            vel_loss = velocity_loss(y_true, y_pred)
            
            # Combined: prioritize correlation, then velocity, then MSE
            return 0.5 * corr_loss + 0.3 * vel_loss + 0.2 * mse
        
        # Learning rate schedule with warmup
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate * 0.5,  # Start lower
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        
        # Compile with trajectory-focused loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=trajectory_combined_loss,  # Correlation-aware loss
            metrics=['mae', 'mse']  # Mean Absolute Error and MSE for monitoring
        )
    
    def build(self) -> keras.Model:
        """Get the built model."""
        if self.model is None:
            self._build_architecture()
        return self.model
    
    def get_model(self) -> keras.Model:
        """Get the compiled model."""
        return self.build()
    
    def summary(self):
        """Print detailed model summary."""
        if self.model is None:
            self.build()
        
        print("=" * 80)
        print("🎯 TRAJECTORY-ONLY MODEL ARCHITECTURE")
        print("=" * 80)
        print(f"📊 Input Shape: (batch, {self.sequence_length}, {self.n_features})")
        print(f"🎯 Output Shape: (batch, {self.sequence_length}, {self.coordinate_dim}) - Trajectory Coordinates")
        print(f"🔄 LSTM Units: {self.lstm_units}")
        print(f"🧠 CNN Filters: {self.cnn_filters}")
        print(f"💧 Dropout Rate: {self.dropout_rate}")
        print(f"📈 Learning Rate: {self.learning_rate}")
        print("=" * 80)
        
        self.model.summary()
    
    def predict_trajectory(self, sensor_data: np.ndarray) -> Dict:
        """
        Predict trajectory coordinates from sensor data.
        
        Args:
            sensor_data: Shape (batch, time, features) or (time, features)
            
        Returns:
            Dictionary with trajectory coordinates and metadata
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        # Handle single sequence input
        if len(sensor_data.shape) == 2:
            sensor_data = np.expand_dims(sensor_data, axis=0)
        
        # Predict trajectory
        trajectory_coords = self.model.predict(sensor_data, verbose=0)
        
        # Calculate trajectory metrics
        batch_size = trajectory_coords.shape[0]
        trajectory_length = []
        velocity_stats = []
        
        for i in range(batch_size):
            coords = trajectory_coords[i]
            
            # Calculate trajectory length (path distance)
            diff = np.diff(coords, axis=0)
            distances = np.sqrt(np.sum(diff**2, axis=1))
            total_length = np.sum(distances)
            trajectory_length.append(total_length)
            
            # Calculate velocity statistics
            velocities = np.sqrt(np.sum(diff**2, axis=1))
            velocity_stats.append({
                'mean_velocity': np.mean(velocities),
                'max_velocity': np.max(velocities),
                'velocity_std': np.std(velocities)
            })
        
        return {
            'trajectory_coordinates': trajectory_coords,
            'trajectory_length': trajectory_length,
            'velocity_statistics': velocity_stats,
            'coordinate_range': {
                'x_min': float(np.min(trajectory_coords[:, :, 0])),
                'x_max': float(np.max(trajectory_coords[:, :, 0])),
                'y_min': float(np.min(trajectory_coords[:, :, 1])),
                'y_max': float(np.max(trajectory_coords[:, :, 1]))
            }
        }
    
    def save_model(self, filepath: str, save_metadata: bool = True):
        """Save the trajectory model and metadata."""
        if self.model is None:
            raise ValueError("Model not built yet. Cannot save.")
        
        # Save model
        self.model.save(filepath)
        print(f"✅ Trajectory model saved to: {filepath}")
        
        # Save metadata
        if save_metadata:
            import json
            from pathlib import Path
            
            metadata = {
                'model_type': 'trajectory_only',
                'version': '1.0',
                'architecture': {
                    'sequence_length': self.sequence_length,
                    'n_features': self.n_features,
                    'coordinate_dim': self.coordinate_dim,
                    'lstm_units': self.lstm_units,
                    'cnn_filters': list(self.cnn_filters),
                    'dropout_rate': self.dropout_rate
                },
                'training_config': {
                    'learning_rate': self.learning_rate,
                    'loss_function': 'mse',
                    'metrics': ['mae', 'mse']
                },
                'output': {
                    'trajectory_coordinates': 'Primary output - continuous (x,y) coordinates',
                    'coordinate_range': '[-1, 1] normalized coordinates',
                    'temporal_resolution': 'Per-timestep trajectory points'
                }
            }
            
            metadata_path = Path(filepath).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model metadata saved to: {metadata_path}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TrajectoryOnlyModel':
        """Load a saved trajectory model."""
        import json
        from pathlib import Path
        
        # Load metadata if available
        metadata_path = Path(filepath).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                arch = metadata['architecture']
                config = metadata['training_config']
        else:
            # Default values if no metadata
            arch = {
                'sequence_length': 500, 'n_features': 48, 'coordinate_dim': 2,
                'lstm_units': 64, 'cnn_filters': [64, 32, 16], 'dropout_rate': 0.3
            }
            config = {'learning_rate': 0.001}
        
        # Create instance
        instance = cls(**arch, **config)
        
        # Load model with custom objects
        custom_objects = {
            'SpatialAttentionLayer': SpatialAttentionLayer
        }
        
        instance.model = keras.models.load_model(filepath, custom_objects=custom_objects)
        
        print(f"✅ Trajectory model loaded from: {filepath}")
        return instance



# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
MODEL_FILE = 'trajectory_model.h5'
NUM_CHANNELS = 5

# --- Model & GUI Settings ---
# These MUST match your model's training!
SEQ_LENGTH = 500  # Input shape (500, 46)
NUM_FEATURES = 46 # Input shape (500, 46)

# The 10 raw features we get from the Arduino
NUM_RAW_FEATURES = NUM_CHANNELS * 2 # (Filtered, Error) * 5

# GUI Window
PREVIEW_WIDTH = 600
PREVIEW_HEIGHT = 400
DOT_SIZE = 10

# --- Global Variables ---
ser = None
window = None
canvas = None
dot = None
# Use a deque to automatically keep the last 500 readings
raw_data_buffer = deque(maxlen=SEQ_LENGTH)


def preprocess_data(raw_sequence):
    """
    !!!!!!!!!!  YOU MUST IMPLEMENT THIS !!!!!!!!!!!
    
    This function receives a (500, 10) list of the most recent raw sensor data
    (f0, e0, f1, e1, f2, e2, f3, e3, f4, e4).
    
    You must convert this into a (500, 46) numpy array, using the
    EXACT same feature engineering logic you used to train the model.
    
    (e.g., adding time-delays, velocity, acceleration, etc.)
    
    !!!!!!!!!!  !!!!!!!!!!! !!!!!!!!!!! !!!!!!!!!!!
    """
    
    # --- START OF PLACEHOLDER CODE (DELETE THIS) ---
    # print("ERROR: 'preprocess_data' is not implemented! Using zeros.")
    # This is just a placeholder so the script can run.
    # The model's predictions will be meaningless until you
    # replace this with your actual preprocessing logic.
    processed_array = np.zeros((SEQ_LENGTH, NUM_FEATURES))
    # --- END OF PLACEHOLDER CODE ---
    
    
    # --- EXAMPLE of what your code *might* look like ---
    # (This is just a guess!)
    #
    # processed_array = np.array(raw_sequence) # Shape (500, 10)
    #
    # # Example: Add 3 time-delay features for each raw feature
    # delay_1 = np.roll(processed_array, 1, axis=0)
    # delay_2 = np.roll(processed_array, 2, axis=0)
    # delay_3 = np.roll(processed_array, 3, axis=0)
    #
    # # Example: Add velocity (difference) for each raw feature
    # velocity = np.diff(processed_array, axis=0, prepend=processed_array[0:1,:])
    #
    # # Combine them all... (10 + 10 + 10 + 10 + 6 = 46)
    # # This is just a guess! Your logic is probably different.
    # final_features = np.concatenate([
    #     processed_array,
    #     delay_1,
    #     delay_2,
    #     delay_3,
    #     velocity[:, 0:6] # Just to make it add up to 46
    # ], axis=1)
    #
    # return final_features
    # --- END OF EXAMPLE ---
    
    return processed_array


def read_serial_data():
    """
    Non-blocking function to read from serial and add to the buffer.
    """
    global raw_data_buffer, ser
    
    # Temporary dict to store readings for one cycle
    current_readings = {}
    
    try:
        # Read as many lines as are waiting, to clear the serial buffer
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue

            match = re.search(r"Channel\s*(\d+):\s*(\d+),\s*error\s*:(-?\d+)", line)
            
            if match:
                channel_num = int(match.group(1))
                filtered_val = int(match.group(2))
                error_val = int(match.group(3))
                
                # Store both values
                current_readings[channel_num] = (filtered_val, error_val)

                # If we've collected all 5 channels, add them to the buffer
                if len(current_readings) == NUM_CHANNELS:
                    row_data = []
                    for i in range(NUM_CHANNELS):
                        row_data.extend(current_readings[i]) # Adds (f, e) tuple
                    
                    # Add the new 10-feature row to our buffer
                    # The deque automatically removes the oldest item
                    raw_data_buffer.append(row_data)
                    
                    # Reset for next cycle
                    current_readings = {}
                    
    except Exception as e:
        print(f"Serial read error: {e}")

    # Reschedule this function to run again
    window.after(5, read_serial_data) # Check for new data every 5ms


def update_prediction(model):
    """
    This function runs ~30 times per second.
    It takes the data buffer, preprocesses it, predicts, and moves the dot.
    """
    global raw_data_buffer, canvas, dot
    
    # Wait until the buffer is full before we start predicting
    if len(raw_data_buffer) < SEQ_LENGTH:
        # Schedule the next update
        window.after(33, update_prediction, model) # ~30 FPS
        return

    # 1. Get the raw data sequence (shape will be 500x10)
    raw_sequence = list(raw_data_buffer)
    
    # 2. Preprocess the data (YOU MUST IMPLEMENT THIS)
    # This must turn the (500, 10) list into a (500, 46) numpy array
    processed_sequence = preprocess_data(raw_sequence)
    
    # 3. Reshape for the model
    # Model expects (batch_size, timesteps, features)
    model_input = processed_sequence.reshape(1, SEQ_LENGTH, NUM_FEATURES)
    
    # 4. Predict!
    # The model outputs a sequence of (500, 2) coordinates
    prediction_sequence = model.predict(model_input)
    
    # 5. Get the *last* (most recent) (x, y) coordinate from the prediction
    # Output shape is (1, 500, 2), so we take [0][-1]
    predicted_xy = prediction_sequence[0][-1] # This will be like [0.123, -0.456]
    
    # 6. Scale the output
    # The model's 'tanh' activation [cite: 34] outputs from -1 to 1.
    # We need to scale this to our window coordinates (0 to PREVIEW_WIDTH/HEIGHT).
    x_pos = (predicted_xy[0] + 1) / 2 * PREVIEW_WIDTH
    y_pos = (predicted_xy[1] + 1) / 2 * PREVIEW_HEIGHT
    
    # 7. Move the dot on the canvas
    canvas.coords(
        dot,
        x_pos - DOT_SIZE / 2,
        y_pos - DOT_SIZE / 2,
        x_pos + DOT_SIZE / 2,
        y_pos + DOT_SIZE / 2
    )
    
    # 8. Schedule this function to run again
    window.after(33, update_prediction, model) # ~30 FPS


# --- Main Script ---
try:
    # 1. Load the Keras Model
    print(f"Loading model '{MODEL_FILE}'...")
    # This may require a 'custom_objects' argument if you have custom layers
    # like 'SpatialAttentionLayer'. Let's try without it first.
    try:
        model = tf.keras.models.load_model(
            MODEL_FILE,
            custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer})
    except ValueError as e:
        print(f"Error: Could not load model. It might have custom layers.")
        print(f"Original error: {e}")
        print("You may need to define your custom layers in this script, e.g.:")
        print("model = tf.keras.models.load_model(MODEL_FILE, custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer})")
        exit()
        
    model.summary()
    print("Model loaded successfully.")

    # 2. Connect to Serial Port
    print(f"Attempting to connect to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    print("Connection successful!")
    
    # 3. Set up the GUI
    print("Opening preview window...")
    window = tk.Tk()
    window.title("Live Trajectory Demo")
    window.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
    
    canvas = tk.Canvas(window, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg='black')
    canvas.pack()
    
    # Create the 'dummy mouse' dot at the center
    dot = canvas.create_oval(
        (PREVIEW_WIDTH/2) - DOT_SIZE / 2,
        (PREVIEW_HEIGHT/2) - DOT_SIZE / 2,
        (PREVIEW_WIDTH/2) + DOT_SIZE / 2,
        (PREVIEW_HEIGHT/2) + DOT_SIZE / 2,
        fill='red', 
        outline='red'
    )
    
    print("GUI ready. Waiting for data buffer to fill...")
    
    # 4. Start the loops
    # Start reading serial data in the background
    read_serial_data()
    # Start the prediction/GUI update loop
    update_prediction(model)
    
    # 5. Start the main GUI loop
    window.mainloop()
    

except KeyboardInterrupt:
    print("\nStopping demo.")
except Exception as e:
    print(f"A fatal error occurred: {e}")
finally:
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
    if window:
        window.destroy()
        print("Window closed.")