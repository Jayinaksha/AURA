"""
Dual-Stream Attention Model for gesture recognition.
Handles asynchronous timing between mouse coordinates and sensor readings.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


class AttentionLayer(layers.Layer):
    """Attention mechanism for focusing on relevant time steps."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_weights = None
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        # Compute attention scores
        e = tf.tanh(tf.tensordot(inputs, self.W, axes=1))
        a = tf.nn.softmax(e, axis=1)
        self.attention_weights = a
        
        # Weighted sum
        attended = inputs * a
        return tf.reduce_sum(attended, axis=1)


class DualStreamAttentionModel:
    """
    Dual-Stream Attention Model for gesture recognition.
    
    Stream 1: Capacitive sensors (primary signal)
    Stream 2: Mouse coordinates (optional, for training only)
    """
    
    def __init__(self, 
                 sequence_length: int = 400,
                 n_sensor_features: int = 10,
                 n_mouse_features: int = 2,
                 n_classes: int = 13,
                 use_mouse_stream: bool = False,
                 lstm_units: int = 128,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        Initialize model.
        
        Args:
            sequence_length: Fixed sequence length
            n_sensor_features: Number of sensor features (10)
            n_mouse_features: Number of mouse features (2)
            n_classes: Number of gesture classes
            use_mouse_stream: Whether to use mouse coordinates stream
            lstm_units: Number of LSTM units per layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_sensor_features = n_sensor_features
        self.n_mouse_features = n_mouse_features
        self.n_classes = n_classes
        self.use_mouse_stream = use_mouse_stream
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
    
    def build(self) -> keras.Model:
        """Build the dual-stream model."""
        # Input: Sensor stream (primary)
        sensor_input = layers.Input(
            shape=(self.sequence_length, self.n_sensor_features),
            name='sensor_input'
        )
        
        # Stream 1: Sensor processing with bidirectional LSTM
        sensor_lstm1 = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(sensor_input)
        sensor_dropout1 = layers.Dropout(self.dropout_rate)(sensor_lstm1)
        
        sensor_lstm2 = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(sensor_dropout1)
        sensor_dropout2 = layers.Dropout(self.dropout_rate)(sensor_lstm2)
        
        # Attention on sensor stream
        sensor_attention = AttentionLayer(name='sensor_attention')(sensor_dropout2)
        
        # Stream 2: Mouse coordinates (optional)
        if self.use_mouse_stream:
            mouse_input = layers.Input(
                shape=(self.sequence_length, self.n_mouse_features),
                name='mouse_input'
            )
            
            # Temporal CNN for mouse coordinates
            mouse_conv1 = layers.Conv1D(filters=32, kernel_size=5, activation='relu')(mouse_input)
            mouse_conv2 = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(mouse_conv1)
            mouse_pool = layers.GlobalMaxPooling1D()(mouse_conv2)
            mouse_dense = layers.Dense(64, activation='relu')(mouse_pool)
            
            # Fusion layer
            fusion = layers.Concatenate()([sensor_attention, mouse_dense])
        else:
            fusion = sensor_attention
        
        # Classification head
        dense1 = layers.Dense(128, activation='relu')(fusion)
        dropout3 = layers.Dropout(self.dropout_rate)(dense1)
        dense2 = layers.Dense(64, activation='relu')(dropout3)
        output = layers.Dense(self.n_classes, activation='softmax', name='gesture_output')(dense2)
        
        # Create model
        if self.use_mouse_stream:
            self.model = keras.Model(
                inputs=[sensor_input, mouse_input],
                outputs=output,
                name='dual_stream_attention_model'
            )
        else:
            self.model = keras.Model(
                inputs=sensor_input,
                outputs=output,
                name='dual_stream_attention_model'
            )
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model(self) -> keras.Model:
        """Get the built model."""
        if self.model is None:
            self.build()
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build()
        self.model.summary()

