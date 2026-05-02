"""
Transformer-based model with relative position encoding.
Robust to absolute timing variations.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class RelativePositionEncoding(layers.Layer):
    """Relative position encoding for time-series."""
    
    def __init__(self, d_model: int, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
    
    def build(self, input_shape):
        # Create relative position matrix
        seq_len = input_shape[1]
        pos = np.arange(seq_len)[:, None]
        rel_pos = pos - pos.T
        
        # Encode relative positions
        self.rel_pos_encoding = self.add_weight(
            name='rel_pos_encoding',
            shape=(self.max_len * 2 + 1, self.d_model),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos = tf.range(seq_len, dtype=tf.float32)
        pos_i = tf.expand_dims(pos, 1)
        pos_j = tf.expand_dims(pos, 0)
        rel_pos = tf.cast(pos_i - pos_j, tf.int32)
        
        # Clip to valid range
        rel_pos = tf.clip_by_value(rel_pos + self.max_len, 0, 2 * self.max_len)
        
        # Get encodings
        rel_encoding = tf.gather(self.rel_pos_encoding, rel_pos)
        
        return inputs + rel_encoding


class TransformerBlock(layers.Layer):
    """Transformer encoder block with multi-head attention."""
    
    def __init__(self, d_model: int, num_heads: int, dff: int, 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # Self-attention
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerGestureModel:
    """
    Transformer model with relative position encoding.
    Robust to timing variations and asynchronous data.
    """
    
    def __init__(self,
                 sequence_length: int = 400,
                 n_features: int = 12,
                 n_classes: int = 13,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dff: int = 512,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 0.001):
        """
        Initialize transformer model.
        
        Args:
            sequence_length: Fixed sequence length
            n_features: Number of input features (12: 10 sensor + 2 mouse)
            n_classes: Number of gesture classes
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dff: Feed-forward network dimension
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
    
    def build(self) -> keras.Model:
        """Build the transformer model."""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Project input to d_model dimension
        x = layers.Dense(self.d_model)(inputs)
        
        # Relative position encoding
        x = RelativePositionEncoding(self.d_model, max_len=self.sequence_length)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Transformer encoder layers
        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.d_model, self.num_heads, self.dff, self.dropout_rate
            )(x)
        
        # Global average pooling (handles variable-length attention)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.n_classes, activation='softmax', 
                              name='gesture_output')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs,
                                name='transformer_gesture_model')
        
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

