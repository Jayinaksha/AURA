"""
SpatialTransformer Model for Capacitive Sensor Gesture Recognition.
Adapted from TraHGR architecture with spatial attention for plate layout.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self attention with spatial awareness."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model, name='query_projection')
        self.wk = layers.Dense(d_model, name='key_projection')
        self.wv = layers.Dense(d_model, name='value_projection')
        
        self.dense = layers.Dense(d_model, name='output_projection')
        self.dropout = layers.Dropout(dropout_rate)
        
    def split_heads(self, x, batch_size):
        """Split last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Compute scaled dot product attention."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale by sqrt(depth)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output


class SpatialPositionalEncoding(layers.Layer):
    """Positional encoding aware of spatial plate layout."""
    
    def __init__(self, d_model: int, max_seq_len: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create positional encoding matrix
        self.pos_encoding = self.get_positional_encoding(max_seq_len, d_model)
        
        # Spatial plate encoding for 5 plates (Left, Upper, Base, Right, Lower)
        self.spatial_encoding = self.get_spatial_encoding(d_model)
        
    def get_positional_encoding(self, seq_len: int, d_model: int):
        """Generate sinusoidal positional encoding."""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.cast(pos_encoding[np.newaxis, :, :], tf.float32)
    
    def get_spatial_encoding(self, d_model: int):
        """Generate spatial encoding for plate layout."""
        # Plate positions in 2D space (normalized)
        plate_positions = np.array([
            [0.0, 0.5],   # Left
            [0.5, 1.0],   # Upper
            [0.5, 0.0],   # Base (center bottom)
            [1.0, 0.5],   # Right
            [0.5, 0.5],   # Lower (center)
        ])
        
        # Encode positions using learnable embeddings
        spatial_features = np.zeros((5, d_model))
        for i, (x, y) in enumerate(plate_positions):
            # Use sine/cosine encoding for x, y coordinates
            for j in range(0, d_model // 4, 2):
                freq = 2 ** (j // 2)
                spatial_features[i, j] = np.sin(x * freq)
                spatial_features[i, j + 1] = np.cos(x * freq)
                spatial_features[i, j + d_model // 2] = np.sin(y * freq)
                spatial_features[i, j + 1 + d_model // 2] = np.cos(y * freq)
        
        return tf.cast(spatial_features, tf.float32)
    
    def call(self, inputs, feature_type_ids=None):
        seq_len = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0]
        
        # Add temporal positional encoding
        inputs += self.pos_encoding[:, :seq_len, :]
        
        # For sensor features, add spatial plate encoding
        # Sensor features 0-9 map to plates 0-4 (2 features per plate)
        if inputs.shape[-1] == self.d_model and seq_len == self.max_seq_len:
            # Create plate IDs: [0,0,1,1,2,2,3,3,4,4] for 10 sensor features
            plate_ids = tf.constant([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=tf.int32)
            
            # Expand to batch and sequence dimensions
            plate_ids = tf.tile(tf.expand_dims(plate_ids, 0), [seq_len, 1])  # (seq_len, 10)
            plate_ids = tf.tile(tf.expand_dims(plate_ids, 0), [batch_size, 1, 1])  # (batch_size, seq_len, 10)
            
            # Only apply spatial encoding if we have 10 features (sensor stream)
            n_features = tf.shape(inputs)[-1] // self.d_model * 10  # Approximate original feature count
            if n_features == 10:
                # Get spatial embeddings for each feature
                spatial_emb = tf.nn.embedding_lookup(self.spatial_encoding, plate_ids)  # (batch, seq, 10, d_model)
                # Average across features dimension
                spatial_emb = tf.reduce_mean(spatial_emb, axis=2)  # (batch, seq, d_model)
                inputs += spatial_emb
        
        return inputs


class SpatialTransformerBlock(layers.Layer):
    """Transformer block with spatial awareness."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dff: int = 512, 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.mha = MultiHeadSelfAttention(d_model, num_heads, dropout_rate)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=None, mask=None):
        # Multi-head attention
        attn_output = self.mha(inputs, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class SpatialFusionLayer(layers.Layer):
    """Fusion layer for combining sensor and spatial features."""
    
    def __init__(self, fusion_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.fusion_dim = fusion_dim
        
        # Sensor feature projection
        self.sensor_projection = layers.Dense(fusion_dim, activation='relu', name='sensor_proj')
        
        # Spatial feature projection  
        self.spatial_projection = layers.Dense(fusion_dim, activation='relu', name='spatial_proj')
        
        # Cross-attention between sensor and spatial features
        self.cross_attention = MultiHeadSelfAttention(fusion_dim, num_heads=8, name='cross_attention')
        
        # Final fusion layers
        self.fusion_layers = keras.Sequential([
            layers.Dense(fusion_dim * 2, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(fusion_dim, activation='relu'),
            layers.LayerNormalization()
        ], name='fusion_network')
        
    def call(self, sensor_features, spatial_features, training=None):
        # Project features to common dimension
        sensor_proj = self.sensor_projection(sensor_features)
        spatial_proj = self.spatial_projection(spatial_features)
        
        # Concatenate for cross-attention
        combined = tf.concat([sensor_proj, spatial_proj], axis=1)
        
        # Apply cross-attention
        attended = self.cross_attention(combined, training=training)
        
        # Global average pooling
        pooled = tf.reduce_mean(attended, axis=1)
        
        # Final fusion
        fused = self.fusion_layers(pooled, training=training)
        
        return fused


class SpatialTransformerModel:
    """
    SpatialTransformer Model for Capacitive Sensor Gesture Recognition.
    
    Architecture:
    1. Dual input streams (sensor features + spatial features)
    2. Spatial positional encoding
    3. Parallel transformer blocks
    4. Spatial fusion layer
    5. Classification head
    """
    
    def __init__(self, 
                 sequence_length: int = 500,
                 n_sensor_features: int = 10,
                 n_spatial_features: int = 38,  # From spatial feature engineering
                 n_classes: int = 13,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dff: int = 512,
                 dropout_rate: float = 0.1,
                 learning_rate: float = 1e-4):
        """
        Initialize SpatialTransformer model.
        
        Args:
            sequence_length: Length of input sequences
            n_sensor_features: Number of sensor features (10)
            n_spatial_features: Number of spatial features (38 from feature engineering)
            n_classes: Number of gesture classes
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dff: Feed-forward network dimension
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.n_sensor_features = n_sensor_features
        self.n_spatial_features = n_spatial_features
        self.n_classes = n_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        
    def build(self) -> keras.Model:
        """Build the SpatialTransformer model."""
        
        # Input layers
        sensor_input = layers.Input(
            shape=(self.sequence_length, self.n_sensor_features),
            name='sensor_input'
        )
        
        spatial_input = layers.Input(
            shape=(self.sequence_length, self.n_spatial_features), 
            name='spatial_input'
        )
        
        # Sensor stream processing
        sensor_projected = layers.Dense(self.d_model, name='sensor_embedding')(sensor_input)
        
        # Create feature type IDs for spatial encoding (map features to plates)
        # Sensor features 0-9 map to plates 0-4 (2 features per plate)
        # Handle this in the positional encoding layer to avoid KerasTensor issues
        sensor_encoded = SpatialPositionalEncoding(
            self.d_model, self.sequence_length, name='sensor_pos_encoding'
        )(sensor_projected)
        
        # Sensor transformer blocks
        sensor_stream = sensor_encoded
        for i in range(self.num_layers):
            sensor_stream = SpatialTransformerBlock(
                self.d_model, self.num_heads, self.dff, self.dropout_rate,
                name=f'sensor_transformer_{i}'
            )(sensor_stream)
        
        # Spatial stream processing  
        spatial_projected = layers.Dense(self.d_model, name='spatial_embedding')(spatial_input)
        spatial_encoded = SpatialPositionalEncoding(
            self.d_model, self.sequence_length, name='spatial_pos_encoding'
        )(spatial_projected)
        
        # Spatial transformer blocks
        spatial_stream = spatial_encoded
        for i in range(self.num_layers // 2):  # Fewer layers for spatial features
            spatial_stream = SpatialTransformerBlock(
                self.d_model, self.num_heads, self.dff, self.dropout_rate,
                name=f'spatial_transformer_{i}'
            )(spatial_stream)
        
        # Fusion layer
        fused_features = SpatialFusionLayer(
            fusion_dim=256, name='spatial_fusion'
        )(sensor_stream, spatial_stream)
        
        # Classification head
        classifier = keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(256, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.n_classes, activation='softmax')
        ], name='classifier')
        
        outputs = classifier(fused_features)
        
        # Create model
        self.model = keras.Model(
            inputs=[sensor_input, spatial_input],
            outputs=outputs,
            name='spatial_transformer_model'
        )
        
        # Compile with advanced optimizer
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(),
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
        
    def prepare_inputs(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare inputs for the dual-stream model.
        
        Args:
            X: (n_samples, seq_len, total_features) where total_features = 48
            
        Returns:
            Tuple of (sensor_features, spatial_features)
        """
        # Split features: first 10 are sensor, next 38 are spatial
        sensor_features = X[:, :, :self.n_sensor_features]  # (n_samples, seq_len, 10)
        spatial_features = X[:, :, self.n_sensor_features:self.n_sensor_features + self.n_spatial_features]  # (n_samples, seq_len, 38)
        
        return sensor_features, spatial_features
