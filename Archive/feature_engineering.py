"""
Enhanced feature engineering and normalization for gesture recognition.
Includes spatial-aware features, robust sensor fusion, and outlier detection.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import signal, stats
from typing import Tuple, Dict, List, Optional
import warnings


class FeatureEngineer:
    """Feature engineering and normalization pipeline."""
    
    def __init__(self, use_mouse_coords: bool = True, normalize_mouse: bool = True):
        """
        Initialize feature engineer.
        
        Args:
            use_mouse_coords: Whether to include mouse coordinates (may be unreliable)
            normalize_mouse: Whether to normalize mouse coordinates to [0,1]
        """
        self.use_mouse_coords = use_mouse_coords
        self.normalize_mouse = normalize_mouse
        self.scaler_sensors = StandardScaler()
        self.scaler_mouse = MinMaxScaler()
        self.mouse_resolution = (1440, 1080)  # Max X, Max Y
        
    def extract_sensor_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extract sensor features from sequence.
        
        Args:
            sequence: (seq_len, 12) - 10 sensor + 2 mouse features
            
        Returns:
            sensor_features: (seq_len, 10) - Only sensor features
        """
        return sequence[:, :10]
    
    def extract_mouse_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract mouse coordinate features."""
        return sequence[:, 10:12]
    
    def normalize_sensors(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize sensor features using StandardScaler.
        
        Args:
            X: (n_samples, seq_len, n_features) or (seq_len, n_features)
            fit: Whether to fit the scaler (use True for training)
            
        Returns:
            Normalized sensor features
        """
        # Check if X is empty
        if X.size == 0:
            return X
        
        original_shape = X.shape
        
        # Handle 1D array case (empty or single sample)
        if len(original_shape) == 1:
            if original_shape[0] == 0:
                return X
            # Single sample case
            X = X.reshape(1, -1)
            original_shape = X.shape
        
        # Reshape for scaler (samples * timesteps, features)
        if len(original_shape) == 3:
            # Ensure we have at least 10 features
            n_features = min(10, original_shape[2])
            X_2d = X[:, :, :n_features].reshape(-1, n_features)
        elif len(original_shape) == 2:
            n_features = min(10, original_shape[1])
            X_2d = X[:, :n_features]
        else:
            # Handle edge cases
            return X
        
        # Fit and transform
        if fit:
            X_normalized = self.scaler_sensors.fit_transform(X_2d)
        else:
            X_normalized = self.scaler_sensors.transform(X_2d)
        
        # Reshape back
        if len(original_shape) == 3:
            return X_normalized.reshape(original_shape[0], original_shape[1], 10)
        else:
            return X_normalized
    
    def normalize_mouse_coords(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize mouse coordinates to [0, 1] range.
        
        Args:
            X: (n_samples, seq_len, n_features) or (seq_len, n_features)
            fit: Whether to fit the scaler
            
        Returns:
            Normalized mouse coordinates
        """
        if not self.use_mouse_coords:
            return None
        
        original_shape = X.shape
        
        # Extract mouse features
        if len(original_shape) == 3:
            mouse_2d = X[:, :, 10:12].reshape(-1, 2)
        else:
            mouse_2d = X[:, 10:12]
        
        # Normalize by resolution
        if fit:
            mouse_normalized = self.scaler_mouse.fit_transform(mouse_2d)
        else:
            mouse_normalized = self.scaler_mouse.transform(mouse_2d)
        
        # Reshape back
        if len(original_shape) == 3:
            return mouse_normalized.reshape(original_shape[0], original_shape[1], 2)
        else:
            return mouse_normalized
    
    def compute_derived_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute derived features from sensor data.
        
        Args:
            sequence: (seq_len, 12) normalized features
            
        Returns:
            derived_features: (seq_len, n_derived) additional features
        """
        sensor_features = self.extract_sensor_features(sequence)
        mouse_features = self.extract_mouse_features(sequence)
        
        derived = []
        
        # Channel magnitudes (strength per channel)
        for i in range(0, 10, 2):  # For each channel (filtered, error)
            filtered = sensor_features[:, i]
            error = sensor_features[:, i + 1]
            magnitude = np.sqrt(filtered**2 + error**2)
            derived.append(magnitude.reshape(-1, 1))
        
        # Velocity from mouse coordinates (if available)
        if self.use_mouse_coords and len(mouse_features) > 0:
            # Compute velocity
            mouse_delta = np.diff(mouse_features, axis=0)
            # Pad first row
            mouse_delta = np.vstack([mouse_delta[0:1], mouse_delta])
            velocity = np.linalg.norm(mouse_delta, axis=1)
            derived.append(velocity.reshape(-1, 1))
            
            # Compute acceleration
            if len(velocity) > 1:
                accel = np.diff(velocity)
                accel = np.concatenate([[accel[0]], accel])
                derived.append(accel.reshape(-1, 1))
        
        # Channel ratios (relative strengths)
        # Ratio of strongest to weakest channel
        channel_magnitudes = np.array([d.flatten() for d in derived[:5]]).T
        if channel_magnitudes.shape[1] > 0:
            max_mag = np.max(channel_magnitudes, axis=1)
            min_mag = np.min(channel_magnitudes, axis=1)
            ratio = np.divide(max_mag, min_mag + 1e-6)  # Avoid division by zero
            derived.append(ratio.reshape(-1, 1))
        
        if len(derived) > 0:
            return np.hstack(derived)
        else:
            return np.array([])
    
    def process_features(self, X: np.ndarray, y: np.ndarray = None, 
                        fit: bool = True, include_derived: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete feature processing pipeline.
        
        Args:
            X: (n_samples, seq_len, 12) raw features
            y: Optional labels for fitting
            fit: Whether to fit scalers
            include_derived: Whether to add derived features
            
        Returns:
            X_processed: Processed features
            y: Labels (unchanged)
        """
        # Check if X is empty
        if X.size == 0 or len(X) == 0:
            print("WARNING: Empty input array, returning empty arrays")
            if len(X.shape) == 3:
                return np.array([]).reshape(0, X.shape[1], X.shape[2]), y if y is not None else np.array([], dtype=int)
            else:
                return X, y if y is not None else np.array([], dtype=int)
        
        # Normalize sensor features
        X_sensors = self.normalize_sensors(X, fit=fit)
        
        # Normalize mouse coordinates (optional)
        X_mouse = self.normalize_mouse_coords(X, fit=fit) if self.use_mouse_coords else None
        
        # Combine features
        if X_mouse is not None:
            X_processed = np.concatenate([X_sensors, X_mouse], axis=-1)
        else:
            X_processed = X_sensors
        
        # Add derived features if requested
        if include_derived:
            derived_features = []
            for i in range(X.shape[0]):
                derived = self.compute_derived_features(X[i])
                if len(derived) > 0:
                    derived_features.append(derived)
            
            if len(derived_features) > 0:
                derived_array = np.array(derived_features)
                X_processed = np.concatenate([X_processed, derived_array], axis=-1)
        
        return X_processed, y


class SpatialPlateFeatureEngineer:
    """
    Spatial-aware feature engineering based on plate layout.
    
    Plate Layout:
    Channel 0: Left
    Channel 1: Upper  
    Channel 2: Base (cleaned/corrected)
    Channel 3: Right
    Channel 4: Lower
    """
    
    def __init__(self):
        self.plate_positions = {
            0: 'Left', 1: 'Upper', 2: 'Base', 3: 'Right', 4: 'Lower'
        }
        
        # Spatial adjacency and distances
        self.adjacency_matrix = np.array([
            [0, 1, 1, 0, 1],  # Left connects to Upper, Base, Lower
            [1, 0, 1, 1, 0],  # Upper connects to Left, Base, Right
            [1, 1, 0, 1, 1],  # Base connects to all (central position)
            [0, 1, 1, 0, 1],  # Right connects to Upper, Base, Lower
            [1, 0, 1, 1, 0]   # Lower connects to Left, Base, Right
        ])
        
        # Distance weights for spatial operations
        self.spatial_distances = np.array([
            [0.0, 1.0, 0.8, 1.4, 1.0],  # Left distances
            [1.0, 0.0, 0.8, 1.0, 1.4],  # Upper distances
            [0.8, 0.8, 0.0, 0.8, 0.8],  # Base distances (central, equidistant)
            [1.4, 1.0, 0.8, 0.0, 1.0],  # Right distances
            [1.0, 1.4, 0.8, 1.0, 0.0]   # Lower distances
        ])
    
    def compute_spatial_gradients(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute spatial gradients between adjacent plates.
        
        Args:
            sequence: (seq_len, n_features) with sensor data
            
        Returns:
            spatial_gradients: (seq_len, n_gradient_features)
        """
        if sequence.shape[1] < 10:
            return np.array([])
            
        seq_len = sequence.shape[0]
        gradients = []
        
        # Extract filtered values for each channel
        channels = []
        for i in range(5):
            if i * 2 < sequence.shape[1]:
                channels.append(sequence[:, i * 2])  # Channel_i_Filtered
            else:
                channels.append(np.zeros(seq_len))
                
        channels = np.array(channels).T  # (seq_len, 5)
        
        # Compute gradients between adjacent plates
        gradient_pairs = [
            (0, 1, 'Left-Upper'), (1, 3, 'Upper-Right'), 
            (3, 4, 'Right-Lower'), (4, 0, 'Lower-Left'),
            (0, 2, 'Left-Base'), (1, 2, 'Upper-Base'),
            (2, 3, 'Base-Right'), (2, 4, 'Base-Lower')
        ]
        
        for ch1, ch2, name in gradient_pairs:
            gradient = channels[:, ch2] - channels[:, ch1]
            gradients.append(gradient)
            
        return np.column_stack(gradients) if gradients else np.array([])
    
    def compute_activation_patterns(self, sequence: np.ndarray, 
                                  activation_threshold: float = 0.1) -> np.ndarray:
        """
        Compute plate activation patterns and sequences.
        
        Args:
            sequence: (seq_len, n_features)
            activation_threshold: Relative threshold for activation detection
            
        Returns:
            activation_features: (seq_len, n_activation_features)
        """
        if sequence.shape[1] < 10:
            return np.array([])
            
        seq_len = sequence.shape[0]
        
        # Extract and normalize channel activities
        channels = []
        for i in range(5):
            if i * 2 < sequence.shape[1]:
                ch_vals = sequence[:, i * 2]  # Channel_i_Filtered
                # Normalize to [0, 1] range for activation detection
                ch_min, ch_max = np.min(ch_vals), np.max(ch_vals)
                if ch_max > ch_min:
                    ch_norm = (ch_vals - ch_min) / (ch_max - ch_min)
                else:
                    ch_norm = np.zeros_like(ch_vals)
                channels.append(ch_norm)
            else:
                channels.append(np.zeros(seq_len))
                
        channels = np.array(channels).T  # (seq_len, 5)
        
        # Activation features
        features = []
        
        # 1. Binary activation masks
        activation_masks = channels > activation_threshold
        features.extend([mask.astype(float) for mask in activation_masks.T])
        
        # 2. Number of active plates at each time
        n_active = np.sum(activation_masks, axis=1)
        features.append(n_active)
        
        # 3. Activation centroid (weighted center of mass)
        plate_positions = np.array([[0, 1], [1, 0], [0.5, 0.5], [1, 1], [0, 1]])  # Approximate positions
        for dim in range(2):  # x, y coordinates
            if np.sum(channels) > 0:
                centroid = np.sum(channels * plate_positions[:, dim], axis=1) / (np.sum(channels, axis=1) + 1e-6)
            else:
                centroid = np.full(seq_len, 0.5)
            features.append(centroid)
        
        # 4. Activation spread (how dispersed the activation is)
        mean_activity = np.mean(channels, axis=1)
        activation_spread = np.std(channels, axis=1)
        features.extend([mean_activity, activation_spread])
        
        return np.column_stack(features) if features else np.array([])
    
    def compute_directional_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute directional movement features based on spatial layout.
        
        Args:
            sequence: (seq_len, n_features)
            
        Returns:
            directional_features: (seq_len, n_directional_features)
        """
        if sequence.shape[1] < 12:  # Need mouse coordinates
            return np.array([])
            
        seq_len = sequence.shape[0]
        mouse_x = sequence[:, 10]
        mouse_y = sequence[:, 11]
        
        features = []
        
        # 1. Mouse velocity components
        dx = np.gradient(mouse_x)
        dy = np.gradient(mouse_y)
        velocity_magnitude = np.sqrt(dx**2 + dy**2)
        
        features.extend([dx, dy, velocity_magnitude])
        
        # 2. Mouse acceleration
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        acceleration_magnitude = np.sqrt(d2x**2 + d2y**2)
        
        features.extend([d2x, d2y, acceleration_magnitude])
        
        # 3. Movement direction (angle)
        movement_angle = np.arctan2(dy, dx + 1e-6)
        features.append(movement_angle)
        
        # 4. Curvature (how much the direction changes)
        angle_change = np.gradient(movement_angle)
        # Handle angle wrapping
        angle_change = np.mod(angle_change + np.pi, 2*np.pi) - np.pi
        features.append(angle_change)
        
        return np.column_stack(features) if features else np.array([])
    
    def compute_sensor_quality_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute features indicating sensor quality and reliability.
        
        Args:
            sequence: (seq_len, n_features)
            
        Returns:
            quality_features: (seq_len, n_quality_features)
        """
        if sequence.shape[1] < 10:
            return np.array([])
            
        seq_len = sequence.shape[0]
        features = []
        
        # For each sensor channel
        for i in range(5):
            if i * 2 + 1 < sequence.shape[1]:
                filtered_vals = sequence[:, i * 2]
                error_vals = sequence[:, i * 2 + 1]
                
                # Signal-to-noise ratio approximation
                signal_power = np.abs(filtered_vals) + 1e-6
                noise_power = np.abs(error_vals) + 1e-6
                snr = signal_power / noise_power
                
                # Signal stability (inverse of local variance)
                window_size = 5
                rolling_std = np.array([
                    np.std(filtered_vals[max(0, j-window_size):j+window_size+1])
                    for j in range(seq_len)
                ])
                stability = 1.0 / (rolling_std + 1e-6)
                
                features.extend([snr, stability])
        
        return np.column_stack(features) if features else np.array([])


class RobustFeatureEngineer:
    """Enhanced feature engineer with outlier detection and robust processing."""
    
    def __init__(self, use_mouse_coords: bool = True, normalize_mouse: bool = True,
                 enable_spatial_features: bool = True, outlier_detection: bool = True):
        """
        Initialize robust feature engineer.
        
        Args:
            use_mouse_coords: Whether to include mouse coordinates
            normalize_mouse: Whether to normalize mouse coordinates
            enable_spatial_features: Whether to compute spatial plate features
            outlier_detection: Whether to apply outlier detection and correction
        """
        self.use_mouse_coords = use_mouse_coords
        self.normalize_mouse = normalize_mouse
        self.enable_spatial_features = enable_spatial_features
        self.outlier_detection = outlier_detection
        
        # Scalers - use RobustScaler for better outlier handling
        self.scaler_sensors = RobustScaler()
        self.scaler_mouse = MinMaxScaler()
        self.scaler_spatial = StandardScaler()
        
        # Spatial feature engineer
        self.spatial_engineer = SpatialPlateFeatureEngineer()
        
        # Fitted flags
        self.is_fitted = False
        
    def detect_and_correct_outliers(self, X: np.ndarray, method: str = 'iqr',
                                   iqr_factor: float = 2.0) -> np.ndarray:
        """
        Detect and correct outliers using robust methods.
        
        Args:
            X: Input features (any shape)
            method: 'iqr' or 'zscore'
            iqr_factor: Factor for IQR-based outlier detection
            
        Returns:
            Corrected features
        """
        if not self.outlier_detection or X.size == 0:
            return X
            
        X_corrected = X.copy()
        original_shape = X.shape
        
        # Flatten to 2D for processing
        if len(original_shape) > 2:
            X_2d = X.reshape(-1, original_shape[-1])
        else:
            X_2d = X.copy()
            
        # Process each feature column
        for i in range(X_2d.shape[1]):
            feature_vals = X_2d[:, i]
            
            if method == 'iqr':
                Q1 = np.percentile(feature_vals, 25)
                Q3 = np.percentile(feature_vals, 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - iqr_factor * IQR
                upper_bound = Q3 + iqr_factor * IQR
                
                outlier_mask = (feature_vals < lower_bound) | (feature_vals > upper_bound)
            else:  # zscore
                z_scores = np.abs(stats.zscore(feature_vals))
                outlier_mask = z_scores > 3.0
                
            # Correct outliers using median
            if np.any(outlier_mask):
                median_val = np.median(feature_vals[~outlier_mask])
                X_2d[outlier_mask, i] = median_val
        
        # Reshape back
        return X_2d.reshape(original_shape)
    
    def extract_comprehensive_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive feature set including spatial features.
        
        Args:
            sequence: (seq_len, 12) raw sequence
            
        Returns:
            comprehensive_features: (seq_len, n_total_features)
        """
        features = []
        
        # 1. Basic sensor features (10 channels)
        sensor_features = sequence[:, :10]
        features.append(sensor_features)
        
        # 2. Mouse coordinates (if enabled)
        if self.use_mouse_coords and sequence.shape[1] >= 12:
            mouse_features = sequence[:, 10:12]
            features.append(mouse_features)
        
        # 3. Spatial features (if enabled)
        if self.enable_spatial_features:
            try:
                # Spatial gradients
                spatial_gradients = self.spatial_engineer.compute_spatial_gradients(sequence)
                if spatial_gradients.size > 0:
                    features.append(spatial_gradients)
                
                # Activation patterns
                activation_patterns = self.spatial_engineer.compute_activation_patterns(sequence)
                if activation_patterns.size > 0:
                    features.append(activation_patterns)
                
                # Directional features
                directional_features = self.spatial_engineer.compute_directional_features(sequence)
                if directional_features.size > 0:
                    features.append(directional_features)
                
                # Quality features
                quality_features = self.spatial_engineer.compute_sensor_quality_features(sequence)
                if quality_features.size > 0:
                    features.append(quality_features)
                    
            except Exception as e:
                warnings.warn(f"Spatial feature extraction failed: {e}")
        
        # Combine all features
        if len(features) > 1:
            combined_features = np.column_stack(features)
        else:
            combined_features = features[0]
            
        return combined_features
    
    def process_features(self, X: np.ndarray, y: np.ndarray = None, 
                        fit: bool = True, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process features with outlier detection, spatial features, and robust scaling.
        
        Args:
            X: (n_samples, seq_len, 12) raw features
            y: Optional labels
            fit: Whether to fit scalers
            verbose: Enable logging
            
        Returns:
            X_processed, y
        """
        if X.size == 0 or len(X) == 0:
            if verbose:
                print("WARNING: Empty input array")
            return X, y if y is not None else np.array([], dtype=int)
        
        if verbose:
            print(f"Processing features: {X.shape}")
        
        # Extract comprehensive features for each sequence
        processed_sequences = []
        
        for i in range(len(X)):
            sequence = X[i]
            
            # Extract comprehensive features
            comprehensive_features = self.extract_comprehensive_features(sequence)
            
            # Apply outlier detection and correction
            corrected_features = self.detect_and_correct_outliers(comprehensive_features)
            
            processed_sequences.append(corrected_features)
        
        # Stack sequences
        X_processed = np.array(processed_sequences)
        
        if verbose:
            print(f"Enhanced features shape: {X_processed.shape}")
        
        # Normalize features
        original_shape = X_processed.shape
        n_samples, seq_len, n_features = original_shape
        
        # Separate sensor, mouse, and spatial features for different normalization
        sensor_end = 10
        mouse_end = 12 if self.use_mouse_coords else 10
        
        X_2d = X_processed.reshape(-1, n_features)
        
        # Normalize sensor features (0:10) with RobustScaler
        if fit:
            self.scaler_sensors.fit(X_2d[:, :sensor_end])
        X_2d[:, :sensor_end] = self.scaler_sensors.transform(X_2d[:, :sensor_end])
        
        # Normalize mouse coordinates (10:12) with MinMaxScaler
        if self.use_mouse_coords and mouse_end <= n_features:
            if fit:
                self.scaler_mouse.fit(X_2d[:, sensor_end:mouse_end])
            X_2d[:, sensor_end:mouse_end] = self.scaler_mouse.transform(X_2d[:, sensor_end:mouse_end])
        
        # Normalize spatial features (12+) with StandardScaler
        if mouse_end < n_features:
            if fit:
                self.scaler_spatial.fit(X_2d[:, mouse_end:])
            X_2d[:, mouse_end:] = self.scaler_spatial.transform(X_2d[:, mouse_end:])
        
        # Reshape back
        X_processed = X_2d.reshape(original_shape)
        
        if fit:
            self.is_fitted = True
            
        if verbose:
            print(f"Final processed shape: {X_processed.shape}")
        
        return X_processed, y

