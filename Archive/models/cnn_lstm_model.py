"""
1D CNN + Bidirectional LSTM model with attention pooling.
Simpler alternative that handles variable lengths well.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .dual_stream_model import AttentionLayer


class CNNLSTMModel:
    """
    1D CNN + Bidirectional LSTM model with attention pooling.
    Simpler architecture that works well for gesture recognition.
    """
    
    def __init__(self,
                 sequence_length: int = 400,
                 n_features: int = 12,
                 n_classes: int = 13,
                 lstm_units: int = 128,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        Initialize CNN-LSTM model.
        
        Args:
            sequence_length: Fixed sequence length
            n_features: Number of input features
            n_classes: Number of gesture classes
            lstm_units: Number of LSTM units per layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
    
    def build(self) -> keras.Model:
        """Build the CNN-LSTM model."""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # CNN feature extraction (multiple filter sizes)
        conv1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1 = layers.BatchNormalization()(conv1)
        
        conv2 = layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(conv1)
        conv2 = layers.BatchNormalization()(conv2)
        
        conv3 = layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(conv2)
        conv3 = layers.BatchNormalization()(conv3)
        
        # Concatenate multi-scale features
        conv_concat = layers.Concatenate(axis=-1)([conv1, conv2, conv3])
        conv_pool = layers.MaxPooling1D(pool_size=2)(conv_concat)
        
        # Bidirectional LSTM
        lstm1 = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(conv_pool)
        dropout1 = layers.Dropout(self.dropout_rate)(lstm1)
        
        lstm2 = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(dropout1)
        dropout2 = layers.Dropout(self.dropout_rate)(lstm2)
        
        # Attention pooling (handles variable-length sequences)
        attention = AttentionLayer(name='attention_pooling')(dropout2)
        
        # Classification head
        dense1 = layers.Dense(128, activation='relu')(attention)
        dropout3 = layers.Dropout(self.dropout_rate)(dense1)
        dense2 = layers.Dense(64, activation='relu')(dropout3)
        outputs = layers.Dense(self.n_classes, activation='softmax',
                              name='gesture_output')(dense2)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs,
                                name='cnn_lstm_model')
        
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

