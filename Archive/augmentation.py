"""
Enhanced data augmentation methods for gesture recognition.
Handles time-domain, amplitude-domain, and spatial-domain augmentations.
Includes class balancing and gesture-aware augmentation.
"""

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Optional, List, Dict
from collections import Counter
from tqdm import tqdm


class GestureAugmentation:
    """Data augmentation for time-series gesture data."""
    
    def __init__(self, noise_std: float = 0.02, amplitude_scale_range: Tuple[float, float] = (0.9, 1.1),
                 time_warp_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Initialize augmentation parameters.
        
        Args:
            noise_std: Standard deviation for Gaussian noise (relative to signal range)
            amplitude_scale_range: Range for amplitude scaling (min, max)
            time_warp_range: Range for time warping factor (min, max)
        """
        self.noise_std = noise_std
        self.amplitude_scale_range = amplitude_scale_range
        self.time_warp_range = time_warp_range
    
    def add_gaussian_noise(self, sequence: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to sequence.
        
        Args:
            sequence: (seq_len, n_features)
            
        Returns:
            Augmented sequence with noise
        """
        signal_range = np.max(sequence) - np.min(sequence)
        noise = np.random.normal(0, self.noise_std * signal_range, sequence.shape)
        return sequence + noise
    
    def amplitude_scaling(self, sequence: np.ndarray) -> np.ndarray:
        """
        Scale amplitude of sensor features.
        
        Args:
            sequence: (seq_len, n_features)
            
        Returns:
            Scaled sequence
        """
        scale_factor = np.random.uniform(*self.amplitude_scale_range)
        # Only scale sensor features (first 10), not mouse coordinates
        augmented = sequence.copy()
        augmented[:, :10] *= scale_factor
        return augmented
    
    def time_warp(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply time warping (stretch/compress) using interpolation.
        
        Args:
            sequence: (seq_len, n_features)
            
        Returns:
            Time-warped sequence (same length)
        """
        seq_len = len(sequence)
        warp_factor = np.random.uniform(*self.time_warp_range)
        
        # Original time indices
        orig_indices = np.arange(seq_len)
        
        # New time indices (warped)
        new_indices = np.linspace(0, seq_len - 1, int(seq_len * warp_factor))
        
        # Interpolate each feature
        warped_sequence = np.zeros_like(sequence)
        for i in range(sequence.shape[1]):
            interp_func = interp1d(orig_indices, sequence[:, i], 
                                 kind='cubic', fill_value='extrapolate')
            warped_values = interp_func(new_indices)
            
            # Resample back to original length
            if len(warped_values) > seq_len:
                # Downsample
                indices = np.linspace(0, len(warped_values) - 1, seq_len).astype(int)
                warped_sequence[:, i] = warped_values[indices]
            elif len(warped_values) < seq_len:
                # Upsample with interpolation
                indices = np.linspace(0, len(warped_values) - 1, seq_len)
                interp_func2 = interp1d(np.arange(len(warped_values)), warped_values,
                                      kind='linear', fill_value='extrapolate')
                warped_sequence[:, i] = interp_func2(indices)
            else:
                warped_sequence[:, i] = warped_values
        
        return warped_sequence
    
    def time_shift(self, sequence: np.ndarray, shift_factor: float = 0.1) -> np.ndarray:
        """
        Apply circular time shift.
        
        Args:
            sequence: (seq_len, n_features)
            shift_factor: Fraction of sequence length to shift
            
        Returns:
            Time-shifted sequence
        """
        shift_amount = int(len(sequence) * shift_factor)
        shift_amount = np.random.randint(-shift_amount, shift_amount + 1)
        return np.roll(sequence, shift_amount, axis=0)
    
    def magnitude_warp(self, sequence: np.ndarray, sigma: float = 2.0, 
                     knot_count: int = 4) -> np.ndarray:
        """
        Apply smooth magnitude warping using random smooth curves.
        
        Args:
            sequence: (seq_len, n_features)
            sigma: Standard deviation of warping magnitude
            knot_count: Number of control points for smooth curve
            
        Returns:
            Magnitude-warped sequence
        """
        seq_len = sequence.shape[0]
        
        # Generate smooth warping curve
        knot_x = np.linspace(0, seq_len - 1, knot_count)
        knot_y = np.random.normal(1.0, sigma * 0.1, knot_count)
        
        # Interpolate to full length
        interp_func = interp1d(knot_x, knot_y, kind='cubic', 
                             fill_value='extrapolate', bounds_error=False)
        warp_curve = interp_func(np.arange(seq_len))
        
        # Apply warping to sensor features only
        augmented = sequence.copy()
        augmented[:, :10] *= warp_curve.reshape(-1, 1)
        
        return augmented
    
    def channel_dropout(self, sequence: np.ndarray, dropout_prob: float = 0.1) -> np.ndarray:
        """
        Randomly zero out sensor channels (handles sensor failures).
        
        Args:
            sequence: (seq_len, n_features)
            dropout_prob: Probability of dropping a channel
            
        Returns:
            Sequence with dropped channels
        """
        augmented = sequence.copy()
        
        # Randomly select channels to drop (sensor channels only)
        n_sensor_channels = 10
        channels_to_drop = np.random.rand(n_sensor_channels) < dropout_prob
        
        for i in range(0, n_sensor_channels, 2):  # Drop filtered and error together
            if channels_to_drop[i]:
                augmented[:, i:i+2] = 0
        
        return augmented
    
    def temporal_subsampling(self, sequence: np.ndarray, 
                           subsample_factor: float = 0.8) -> np.ndarray:
        """
        Temporal subsampling/upsampling (different gesture speeds).
        
        Args:
            sequence: (seq_len, n_features)
            subsample_factor: Factor to subsample (0.8 = slower, 1.2 = faster)
            
        Returns:
            Resampled sequence (same length via interpolation)
        """
        seq_len = len(sequence)
        new_len = int(seq_len * subsample_factor)
        
        # Resample
        orig_indices = np.arange(seq_len)
        new_indices = np.linspace(0, seq_len - 1, new_len)
        
        resampled = np.zeros((new_len, sequence.shape[1]))
        for i in range(sequence.shape[1]):
            interp_func = interp1d(orig_indices, sequence[:, i], 
                                 kind='cubic', fill_value='extrapolate')
            resampled[:, i] = interp_func(new_indices)
        
        # Resample back to original length
        if new_len != seq_len:
            final_indices = np.linspace(0, new_len - 1, seq_len)
            interp_func2 = interp1d(np.arange(new_len), resampled, 
                                  kind='linear', axis=0, fill_value='extrapolate')
            resampled = interp_func2(final_indices)
        
        return resampled
    
    def apply_augmentation(self, sequence: np.ndarray, 
                          augmentation_type: Optional[str] = None) -> np.ndarray:
        """
        Apply random augmentation or specific type.
        
        Args:
            sequence: (seq_len, n_features)
            augmentation_type: Specific augmentation to apply, or None for random
            
        Returns:
            Augmented sequence
        """
        augmentations = [
            self.add_gaussian_noise,
            self.amplitude_scaling,
            self.time_warp,
            self.time_shift,
            self.magnitude_warp,
            self.temporal_subsampling,
        ]
        
        augmentation_names = [
            'noise', 'amplitude', 'time_warp', 'time_shift', 
            'magnitude_warp', 'subsampling'
        ]
        
        if augmentation_type:
            idx = augmentation_names.index(augmentation_type)
            return augmentations[idx](sequence)
        else:
            # Random augmentation
            aug_func = np.random.choice(augmentations)
            return aug_func(sequence)
    
    def augment_batch(self, X: np.ndarray, y: np.ndarray, 
                     augmentation_prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment a batch of sequences.
        
        Args:
            X: (batch_size, seq_len, n_features)
            y: (batch_size,)
            augmentation_prob: Probability of augmenting each sample
            
        Returns:
            Augmented X and y
        """
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            if np.random.rand() < augmentation_prob:
                augmented_seq = self.apply_augmentation(X[i])
                augmented_X.append(augmented_seq)
                augmented_y.append(y[i])
            else:
                augmented_X.append(X[i])
                augmented_y.append(y[i])
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def elastic_transform(self, sequence: np.ndarray, alpha: float = 10.0, 
                         sigma: float = 2.0) -> np.ndarray:
        """
        Apply elastic deformation to sensor data.
        
        Args:
            sequence: (seq_len, n_features)
            alpha: Scaling factor for deformation
            sigma: Standard deviation for Gaussian smoothing
            
        Returns:
            Elastically deformed sequence
        """
        seq_len = sequence.shape[0]
        
        # Generate random deformation field
        dx = np.random.randn(seq_len) * alpha
        dx = gaussian_filter1d(dx, sigma, mode='constant', cval=0)
        
        # Apply deformation to time indices
        indices = np.arange(seq_len) + dx
        indices = np.clip(indices, 0, seq_len - 1)
        
        # Interpolate features at new positions
        deformed = np.zeros_like(sequence)
        for i in range(sequence.shape[1]):
            interp_func = interp1d(np.arange(seq_len), sequence[:, i],
                                 kind='linear', fill_value='extrapolate')
            deformed[:, i] = interp_func(indices)
            
        return deformed
    
    def jitter(self, sequence: np.ndarray, noise_factor: float = 0.03) -> np.ndarray:
        """
        Add random jitter (small random variations).
        
        Args:
            sequence: (seq_len, n_features)
            noise_factor: Scale of jitter relative to signal std
            
        Returns:
            Jittered sequence
        """
        jittered = sequence.copy()
        
        # Apply jitter to sensor channels only (first 10 features)
        for i in range(min(10, sequence.shape[1])):
            signal_std = np.std(sequence[:, i])
            jitter_noise = np.random.normal(0, signal_std * noise_factor, len(sequence))
            jittered[:, i] += jitter_noise
            
        return jittered
    
    def speed_variation(self, sequence: np.ndarray, speed_range: Tuple[float, float] = (0.7, 1.3)) -> np.ndarray:
        """
        Vary gesture speed (more realistic than simple time warping).
        
        Args:
            sequence: (seq_len, n_features)
            speed_range: Min and max speed multipliers
            
        Returns:
            Speed-varied sequence
        """
        speed_factor = np.random.uniform(*speed_range)
        
        # Non-linear speed variation (more realistic)
        seq_len = len(sequence)
        time_indices = np.arange(seq_len)
        
        # Create smooth speed curve
        knots = 5
        knot_x = np.linspace(0, seq_len - 1, knots)
        knot_speeds = np.random.normal(speed_factor, speed_factor * 0.1, knots)
        
        interp_func = interp1d(knot_x, knot_speeds, kind='cubic', fill_value='extrapolate')
        speed_curve = interp_func(time_indices)
        
        # Apply cumulative speed variation
        cumulative_time = np.cumsum(1.0 / speed_curve)
        cumulative_time = cumulative_time * (seq_len - 1) / cumulative_time[-1]
        
        # Interpolate to new time indices
        varied = np.zeros_like(sequence)
        for i in range(sequence.shape[1]):
            interp_func2 = interp1d(time_indices, sequence[:, i], 
                                  kind='cubic', fill_value='extrapolate')
            varied[:, i] = interp_func2(cumulative_time)
            
        return varied
    
    def trajectory_extrapolate(self, sequence: np.ndarray, extend_by: int = 5) -> np.ndarray:
        """
        Extrapolate sequence based on trajectory analysis with quality preservation.
        Best for linear/angular gestures.
        """
        # Find actual sequence length
        actual_len = np.sum(np.any(sequence != 0, axis=1))
        if actual_len < 3:
            return sequence
        
        seq = sequence[:actual_len].copy()
        
        # Only extend short sequences (preserve quality)
        if actual_len >= 40:
            extend_by = min(extend_by, 2)  # Minimal extension for longer sequences
        
        # For sequences with mouse coordinates
        if seq.shape[1] >= 12:
            mouse_x = seq[:, 10]
            mouse_y = seq[:, 11]
            
            # Calculate velocity trend
            dx = np.diff(mouse_x)
            dy = np.diff(mouse_y)
            
            # Smooth velocity if enough points
            if len(dx) > 3:
                dx_smooth = savgol_filter(dx, min(5, len(dx)), 2) if len(dx) >= 5 else dx
                dy_smooth = savgol_filter(dy, min(5, len(dy)), 2) if len(dy) >= 5 else dy
            else:
                dx_smooth = dx
                dy_smooth = dy
            
            # Get final velocity
            last_vx = dx_smooth[-1] if len(dx_smooth) > 0 else 0
            last_vy = dy_smooth[-1] if len(dy_smooth) > 0 else 0
            
            # Create extended sequence
            extended_seq = np.zeros((actual_len + extend_by, seq.shape[1]))
            extended_seq[:actual_len] = seq
            
            # Extrapolate mouse trajectory with damping
            for i in range(extend_by):
                damping = 0.85 ** (i + 1)  # Velocity decay
                new_x = extended_seq[actual_len + i - 1, 10] + last_vx * damping
                new_y = extended_seq[actual_len + i - 1, 11] + last_vy * damping
                extended_seq[actual_len + i, 10] = new_x
                extended_seq[actual_len + i, 11] = new_y
            
            # Extrapolate sensor channels with constraints
            for j in range(10):  # Sensor channels
                if np.std(seq[:, j]) > 0.01:
                    x = np.arange(actual_len)
                    y = seq[:, j]
                    
                    # Cubic spline with natural boundary conditions
                    try:
                        spline = CubicSpline(x, y, bc_type='natural', extrapolate=True)
                        new_x = np.arange(actual_len, actual_len + extend_by)
                        new_y = spline(new_x)
                        
                        # Apply realistic constraints
                        mean_val = np.mean(y)
                        std_val = np.std(y)
                        new_y = np.clip(new_y, mean_val - 2*std_val, mean_val + 2*std_val)
                        
                        # Add decay towards mean
                        for i in range(extend_by):
                            decay = 0.9 ** (i + 1)
                            new_y[i] = new_y[i] * decay + mean_val * (1 - decay)
                        
                        extended_seq[actual_len:, j] = new_y
                    except:
                        # Fallback: constant extrapolation
                        extended_seq[actual_len:, j] = y[-1]
            
            # Pad to original length if needed
            if len(extended_seq) < len(sequence):
                padded = np.zeros_like(sequence)
                padded[:len(extended_seq)] = extended_seq
                return padded
            else:
                # Truncate if longer than original
                result = sequence.copy()
                result[:len(extended_seq)] = extended_seq[:len(sequence)]
                return result
        
        return sequence
    
    def periodic_extrapolate(self, sequence: np.ndarray, extend_by: int = 5) -> np.ndarray:
        """
        Extrapolate assuming periodic motion. Best for circular gestures.
        """
        actual_len = np.sum(np.any(sequence != 0, axis=1))
        if actual_len < 6:
            return sequence
        
        seq = sequence[:actual_len].copy()
        
        # Find cycle pattern
        cycle_len = min(actual_len // 3, 8)  # Conservative cycle length
        if cycle_len < 3:
            return self.trajectory_extrapolate(sequence, extend_by)
        
        pattern = seq[-cycle_len:]
        extended_seq = np.zeros((actual_len + extend_by, seq.shape[1]))
        extended_seq[:actual_len] = seq
        
        # Repeat pattern with small variations
        for i in range(extend_by):
            idx = i % cycle_len
            base_point = pattern[idx]
            
            # Add small variation
            variation = np.random.normal(0, 0.02, len(base_point))
            extended_seq[actual_len + i] = base_point + variation
        
        # Pad to original length
        if len(extended_seq) < len(sequence):
            padded = np.zeros_like(sequence)
            padded[:len(extended_seq)] = extended_seq
            return padded
        else:
            result = sequence.copy()
            result[:len(extended_seq)] = extended_seq[:len(sequence)]
            return result


class SpatialAugmentation:
    """Spatial augmentation for mouse coordinates and spatial gesture features."""
    
    def __init__(self, screen_resolution: Tuple[int, int] = (1440, 1080)):
        """
        Initialize spatial augmentation.
        
        Args:
            screen_resolution: (width, height) of screen coordinate space
        """
        self.screen_width, self.screen_height = screen_resolution
        
    def translate_mouse(self, sequence: np.ndarray, 
                       max_translation: Tuple[float, float] = (0.1, 0.1)) -> np.ndarray:
        """
        Translate mouse coordinates (shift gesture position).
        
        Args:
            sequence: (seq_len, n_features) 
            max_translation: Max translation as fraction of screen size
            
        Returns:
            Sequence with translated mouse coordinates
        """
        if sequence.shape[1] < 12:  # Need mouse coordinates
            return sequence
            
        augmented = sequence.copy()
        
        # Random translation amounts
        tx = np.random.uniform(-max_translation[0], max_translation[0]) * self.screen_width
        ty = np.random.uniform(-max_translation[1], max_translation[1]) * self.screen_height
        
        # Apply translation to mouse coordinates (last 2 columns)
        augmented[:, 10] += tx  # Mouse_X
        augmented[:, 11] += ty  # Mouse_Y
        
        # Ensure coordinates stay within screen bounds
        augmented[:, 10] = np.clip(augmented[:, 10], 0, self.screen_width)
        augmented[:, 11] = np.clip(augmented[:, 11], 0, self.screen_height)
        
        return augmented
    
    def rotate_mouse(self, sequence: np.ndarray, 
                    max_angle: float = 0.15) -> np.ndarray:
        """
        Rotate mouse coordinates around gesture center.
        
        Args:
            sequence: (seq_len, n_features)
            max_angle: Maximum rotation angle in radians
            
        Returns:
            Sequence with rotated mouse coordinates
        """
        if sequence.shape[1] < 12:
            return sequence
            
        augmented = sequence.copy()
        mouse_x = augmented[:, 10]
        mouse_y = augmented[:, 11]
        
        # Find gesture center
        center_x = np.mean(mouse_x)
        center_y = np.mean(mouse_y)
        
        # Random rotation angle
        angle = np.random.uniform(-max_angle, max_angle)
        
        # Rotate around center
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Translate to origin, rotate, translate back
        rel_x = mouse_x - center_x
        rel_y = mouse_y - center_y
        
        rotated_x = rel_x * cos_a - rel_y * sin_a + center_x
        rotated_y = rel_x * sin_a + rel_y * cos_a + center_y
        
        # Ensure coordinates stay within bounds
        augmented[:, 10] = np.clip(rotated_x, 0, self.screen_width)
        augmented[:, 11] = np.clip(rotated_y, 0, self.screen_height)
        
        return augmented
    
    def scale_mouse(self, sequence: np.ndarray,
                   scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Scale mouse gesture around its center.
        
        Args:
            sequence: (seq_len, n_features)
            scale_range: (min_scale, max_scale)
            
        Returns:
            Sequence with scaled mouse coordinates
        """
        if sequence.shape[1] < 12:
            return sequence
            
        augmented = sequence.copy()
        mouse_x = augmented[:, 10]
        mouse_y = augmented[:, 11]
        
        # Find gesture center
        center_x = np.mean(mouse_x)
        center_y = np.mean(mouse_y)
        
        # Random scale factors
        scale_x = np.random.uniform(*scale_range)
        scale_y = np.random.uniform(*scale_range)
        
        # Scale around center
        scaled_x = (mouse_x - center_x) * scale_x + center_x
        scaled_y = (mouse_y - center_y) * scale_y + center_y
        
        # Ensure coordinates stay within bounds
        augmented[:, 10] = np.clip(scaled_x, 0, self.screen_width)
        augmented[:, 11] = np.clip(scaled_y, 0, self.screen_height)
        
        return augmented


class GestureAwareAugmentation:
    """Augmentation that respects gesture semantics and spatial patterns."""
    
    def __init__(self, plate_layout: Optional[Dict] = None):
        """
        Initialize gesture-aware augmentation.
        
        Args:
            plate_layout: Optional plate layout information
        """
        self.plate_positions = plate_layout or {
            0: 'Left', 1: 'Upper', 2: 'Base', 3: 'Right', 4: 'Lower'
        }
        
    def preserve_gesture_topology(self, sequence: np.ndarray, 
                                 gesture_type: str) -> np.ndarray:
        """
        Apply augmentation while preserving gesture topology.
        
        Args:
            sequence: (seq_len, n_features)
            gesture_type: Type of gesture (e.g., 'circle', 'triangle')
            
        Returns:
            Augmented sequence with preserved topology
        """
        # Different augmentation strategies based on gesture type
        if 'circle' in gesture_type.lower():
            return self._augment_circular_gesture(sequence)
        elif 'triangle' in gesture_type.lower():
            return self._augment_angular_gesture(sequence)
        elif 'line' in gesture_type.lower() or 'diagonal' in gesture_type.lower():
            return self._augment_linear_gesture(sequence)
        else:
            return self._augment_generic_gesture(sequence)
    
    def _augment_circular_gesture(self, sequence: np.ndarray) -> np.ndarray:
        """Augment circular gestures preserving roundness."""
        augmented = sequence.copy()
        
        # Apply smooth amplitude variations that preserve circular pattern
        seq_len = len(sequence)
        
        # Create smooth radial scaling
        phase = np.linspace(0, 2 * np.pi, seq_len)
        radial_variation = 1.0 + 0.1 * np.sin(phase * 2) * np.random.uniform(0.5, 1.5)
        
        # Apply to sensor channels
        for i in range(0, min(10, sequence.shape[1]), 2):
            augmented[:, i] *= radial_variation
            
        return augmented
    
    def _augment_angular_gesture(self, sequence: np.ndarray) -> np.ndarray:
        """Augment angular gestures (triangles, rectangles) preserving corners."""
        augmented = sequence.copy()
        
        # Identify corner regions (high curvature)
        if sequence.shape[1] >= 12:  # Has mouse coordinates
            mouse_x = sequence[:, 10]
            mouse_y = sequence[:, 11]
            
            # Compute curvature approximation
            dx = np.gradient(mouse_x)
            dy = np.gradient(mouse_y)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-6)**1.5
            corner_mask = curvature > np.percentile(curvature, 80)
            
            # Apply less smoothing near corners
            sigma = 1.0
            smooth_factor = np.where(corner_mask, 0.3, 1.0)
            
            for i in range(min(10, sequence.shape[1])):
                smoothed = gaussian_filter1d(augmented[:, i], sigma)
                augmented[:, i] = (augmented[:, i] * smooth_factor + 
                                 smoothed * (1 - smooth_factor))
        
        return augmented
    
    def _augment_linear_gesture(self, sequence: np.ndarray) -> np.ndarray:
        """Augment linear gestures preserving straightness."""
        augmented = sequence.copy()
        
        # Apply smooth variations along the line direction
        if sequence.shape[1] >= 12:
            mouse_x = sequence[:, 10]
            mouse_y = sequence[:, 11]
            
            # Find line direction
            line_dir = np.array([mouse_x[-1] - mouse_x[0], mouse_y[-1] - mouse_y[0]])
            line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-6)
            
            # Perpendicular direction
            perp_dir = np.array([-line_dir[1], line_dir[0]])
            
            # Add small perpendicular variations
            seq_len = len(sequence)
            perp_variation = np.random.normal(0, 0.02, seq_len)
            
            for i in range(min(10, sequence.shape[1])):
                variation = perp_variation * np.random.uniform(0.5, 1.5)
                augmented[:, i] += variation
        
        return augmented
    
    def _augment_generic_gesture(self, sequence: np.ndarray) -> np.ndarray:
        """Generic augmentation for complex gestures."""
        augmented = sequence.copy()
        
        # Apply gentle smoothing with random variations
        sigma = np.random.uniform(0.5, 2.0)
        
        for i in range(min(10, sequence.shape[1])):
            if np.std(sequence[:, i]) > 0:
                smoothed = gaussian_filter1d(augmented[:, i], sigma)
                blend_factor = np.random.uniform(0.1, 0.3)
                augmented[:, i] = (1 - blend_factor) * augmented[:, i] + blend_factor * smoothed
        
        return augmented


class ClassBalancingAugmentation:
    """Augmentation system for handling class imbalance."""
    
    def __init__(self, target_samples_per_class: Optional[int] = None):
        """
        Initialize class balancing augmentation.
        
        Args:
            target_samples_per_class: Target number of samples per class
        """
        self.target_samples_per_class = target_samples_per_class
        self.base_augmenter = GestureAugmentation()
        self.spatial_augmenter = SpatialAugmentation()
        self.gesture_augmenter = GestureAwareAugmentation()
        
    def balance_dataset(self, X: np.ndarray, y: np.ndarray, 
                       label_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset using targeted augmentation.
        
        Args:
            X: (n_samples, seq_len, n_features)
            y: (n_samples,) encoded labels
            label_names: List of class names
            
        Returns:
            Balanced X and y
        """
        print(f"⚖️ Balancing dataset with {len(X)} samples...")
        
        # Analyze class distribution
        class_counts = Counter(y)
        
        if self.target_samples_per_class is None:
            # Target = max class count
            self.target_samples_per_class = max(class_counts.values())
        
        print(f"Target samples per class: {self.target_samples_per_class}")
        
        balanced_X = []
        balanced_y = []
        
        for class_idx in tqdm(range(len(label_names)), desc="⚖️ Balancing classes", unit="class"):
            class_name = label_names[class_idx]
            current_count = class_counts.get(class_idx, 0)
            
            if current_count == 0:
                print(f"  {class_name}: No samples found, skipping")
                continue
                
            # Get samples for this class
            class_mask = y == class_idx
            class_samples = X[class_mask]
            
            # Add original samples
            balanced_X.extend(class_samples)
            balanced_y.extend([class_idx] * len(class_samples))
            
            # Generate augmented samples if needed
            needed_samples = self.target_samples_per_class - current_count
            
            if needed_samples > 0:
                print(f"  {class_name}: {current_count} -> {self.target_samples_per_class} "
                      f"(+{needed_samples} augmented)")
                
                augmented_samples = self._generate_augmented_samples(
                    class_samples, needed_samples, class_name
                )
                
                balanced_X.extend(augmented_samples)
                balanced_y.extend([class_idx] * len(augmented_samples))
            else:
                print(f"  {class_name}: {current_count} samples (sufficient)")
        
        # Convert to arrays and shuffle
        balanced_X = np.array(balanced_X)
        balanced_y = np.array(balanced_y)
        
        # Shuffle the dataset
        shuffle_idx = np.random.permutation(len(balanced_X))
        balanced_X = balanced_X[shuffle_idx]
        balanced_y = balanced_y[shuffle_idx]
        
        print(f"Balanced dataset: {len(balanced_X)} total samples")
        
        return balanced_X, balanced_y
    
    def _generate_augmented_samples(self, class_samples: np.ndarray, 
                                  needed_samples: int, class_name: str) -> List[np.ndarray]:
        """Generate augmented samples for a specific class."""
        augmented = []
        
        # Quality-focused augmentation strategies
        augmentation_strategies = [
            ('noise', lambda x: self.base_augmenter.add_gaussian_noise(x)),
            ('amplitude', lambda x: self.base_augmenter.amplitude_scaling(x)),
            ('time_warp', lambda x: self.base_augmenter.time_warp(x)),
            ('jitter', lambda x: self.base_augmenter.jitter(x)),
            ('speed', lambda x: self.base_augmenter.speed_variation(x)),
            ('trajectory_extrapolate', lambda x: self.base_augmenter.trajectory_extrapolate(x)),
            ('periodic_extrapolate', lambda x: self.base_augmenter.periodic_extrapolate(x)),
            ('spatial_translate', lambda x: self.spatial_augmenter.translate_mouse(x)),
            ('spatial_rotate', lambda x: self.spatial_augmenter.rotate_mouse(x)),
            ('gesture_aware', lambda x: self.gesture_augmenter.preserve_gesture_topology(x, class_name))
        ]
        
        for i in range(needed_samples):
            # Select random original sample
            orig_sample = class_samples[np.random.randint(len(class_samples))]
            
            # Apply multiple augmentations (stacked)
            augmented_sample = orig_sample.copy()
            
            # Randomly select and apply 1-3 augmentations
            n_augmentations = np.random.randint(1, 4)
            selected_augs = np.random.choice(len(augmentation_strategies), 
                                           n_augmentations, replace=False)
            
            for aug_idx in selected_augs:
                try:
                    aug_name, aug_func = augmentation_strategies[aug_idx]
                    augmented_sample = aug_func(augmented_sample)
                except Exception as e:
                    # If augmentation fails, use original
                    print(f"    Warning: {aug_name} augmentation failed: {e}")
                    continue
            
            augmented.append(augmented_sample)
        
        return augmented

