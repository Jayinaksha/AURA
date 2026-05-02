"""
Data loading and preprocessing for Aura gesture recognition system.
Handles CSV file loading, sequence extraction, and label assignment.
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Tuple, Dict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
try:
    from .data_quality import clean_gesture_data
except ImportError:
    from data_quality import clean_gesture_data


class GestureDataProcessor:
    """Processes raw CSV files into sequences for model training."""
    
    def __init__(self, data_dir: str = ".", min_sequence_length: int = 20, 
                 max_sequence_length: int = 500, gap_threshold_ms: float = 2000.0,
                 adaptive_sequence_detection: bool = True, enable_data_cleaning: bool = True, 
                 verbose: bool = True):
        """
        Initialize data processor.
        
        Args:
            data_dir: Directory containing CSV files
            min_sequence_length: Minimum sequence length to keep (optimized to 20)
            max_sequence_length: Maximum sequence length (will pad/truncate)
            gap_threshold_ms: Time gap (ms) to split sequences (optimal at 2000)
            adaptive_sequence_detection: Use sensor activity to detect sequences
            enable_data_cleaning: Apply comprehensive data quality cleaning
            verbose: Enable detailed logging
        """
        self.data_dir = data_dir
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.gap_threshold_ms = gap_threshold_ms
        self.adaptive_sequence_detection = adaptive_sequence_detection
        self.enable_data_cleaning = enable_data_cleaning
        self.verbose = verbose
        self.label_encoder = LabelEncoder()
        self.gesture_mapping = self._create_gesture_mapping()
        
        # Statistics for monitoring
        self.processing_stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_raw_samples': 0,
            'sequences_extracted': 0,
            'sequences_too_short': 0,
            'large_gaps_found': 0
        }
        
    def _create_gesture_mapping(self) -> Dict[str, str]:
        """Create mapping from filenames to gesture labels."""
        mapping = {
            'circle_reverse.csv': 'circle_reverse',
            'CIRCLE.csv': 'circle',
            'diagonal_left_to_right.csv': 'diagonal_left_to_right',
            'diagonal_right_to_left.csv': 'diagonal_right_to_left',
            'diamond.csv': 'diamond',
            'infinity_vertical.csv': 'infinity_vertical',
            'Infinity.csv': 'infinity',
            'lefttorightline.csv': 'lefttorightline',
            'm_PATTERN.csv': 'double_triangle',
            'rectangle.csv': 'rectangle',
            'triangle_inverted.csv': 'triangle_inverted',
            'triangle_right.csv': 'triangle_right',
            'triangle.csv': 'triangle'
        }
        return mapping
    
    def _infer_label_from_filename(self, filename: str) -> str:
        """Infer gesture label from filename."""
        basename = os.path.basename(filename)
        return self.gesture_mapping.get(basename, basename.replace('.csv', '').lower())
    
    def load_csv_file(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess a single CSV file."""
        df = pd.read_csv(filepath)
        
        # Add label if missing
        if 'Label' not in df.columns:
            label = self._infer_label_from_filename(filepath)
            df['Label'] = label
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Calculate time differences
        df['Time_diff_ms'] = df['Timestamp'].diff().dt.total_seconds() * 1000
        df['Time_diff_ms'] = df['Time_diff_ms'].fillna(0)
        
        return df
    
    def _detect_sensor_activity(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Detect periods of sensor activity for adaptive sequence extraction.
        Uses sensor variance and mouse movement to identify gesture periods.
        
        Returns:
            List of (start_idx, end_idx) tuples for active periods
        """
        # Calculate sensor activity indicators
        sensor_cols = ['Channel_0_Filtered', 'Channel_1_Filtered', 'Channel_3_Filtered', 'Channel_4_Filtered']
        
        # Use rolling window to smooth activity detection
        window_size = 10
        activity_indicators = []
        
        for col in sensor_cols:
            if col in df.columns:
                # Calculate rolling standard deviation
                rolling_std = df[col].rolling(window=window_size, min_periods=1).std()
                activity_indicators.append(rolling_std)
        
        # Mouse movement indicator
        if 'Mouse_X' in df.columns and 'Mouse_Y' in df.columns:
            mouse_velocity = np.sqrt(df['Mouse_X'].diff()**2 + df['Mouse_Y'].diff()**2)
            mouse_velocity = mouse_velocity.fillna(0)
            mouse_activity = mouse_velocity.rolling(window=window_size, min_periods=1).mean()
            activity_indicators.append(mouse_activity)
        
        # Combine activity indicators
        if activity_indicators:
            combined_activity = np.mean(activity_indicators, axis=0)
        else:
            return [(0, len(df))]  # Fallback: treat entire file as one sequence
        
        # Find activity periods (above threshold)
        activity_threshold = np.percentile(combined_activity, 60)  # Top 40% activity
        active_mask = combined_activity > activity_threshold
        
        # Find continuous active regions
        active_periods = []
        in_period = False
        start_idx = 0
        
        for i, is_active in enumerate(active_mask):
            if is_active and not in_period:
                start_idx = i
                in_period = True
            elif not is_active and in_period:
                active_periods.append((start_idx, i))
                in_period = False
        
        # Handle case where period extends to end
        if in_period:
            active_periods.append((start_idx, len(df)))
        
        return active_periods

    def extract_sequences(self, df: pd.DataFrame, filepath: str = "") -> List[Tuple[np.ndarray, str]]:
        """
        Extract gesture sequences from dataframe.
        Uses both time gaps and adaptive sensor activity detection.
        
        Args:
            df: DataFrame with gesture data
            filepath: Optional filepath for logging
            
        Returns:
            List of (sequence_array, label) tuples
        """
        sequences = []
        filename = os.path.basename(filepath) if filepath else "unknown"
        
        if self.verbose:
            print(f"  Processing {filename}: {len(df)} samples")
        
        # Method 1: Time-based splitting (primary method)
        split_points = [0]
        gaps_found = 0
        for i in range(1, len(df)):
            if df.iloc[i]['Time_diff_ms'] > self.gap_threshold_ms:
                split_points.append(i)
                gaps_found += 1
        split_points.append(len(df))
        
        self.processing_stats['large_gaps_found'] += gaps_found
        
        if self.verbose:
            print(f"    Time-based splits: {len(split_points)-1} segments, {gaps_found} large gaps")
        
        # Method 2: Adaptive sequence detection (if enabled and few splits found)
        if self.adaptive_sequence_detection and len(split_points) <= 2:
            if self.verbose:
                print(f"    Using adaptive detection (few time splits found)")
            active_periods = self._detect_sensor_activity(df)
            
            # Merge with time-based splits
            all_periods = []
            for start_time, end_time in zip(split_points[:-1], split_points[1:]):
                # Find active periods within this time segment
                for act_start, act_end in active_periods:
                    overlap_start = max(start_time, act_start)
                    overlap_end = min(end_time, act_end)
                    if overlap_end > overlap_start + self.min_sequence_length:
                        all_periods.append((overlap_start, overlap_end))
            
            # Use adaptive periods if they give more sequences
            if len(all_periods) > len(split_points) - 1:
                split_points = [0] + [end for _, end in all_periods]
                if self.verbose:
                    print(f"    Adaptive detection found {len(all_periods)} additional sequences")
        
        # Extract sequences between split points
        sequences_too_short = 0
        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            sequence_df = df.iloc[start_idx:end_idx].copy()
            
            # Filter out sequences that are too short
            if len(sequence_df) < self.min_sequence_length:
                sequences_too_short += 1
                continue
            
            # Get label (should be consistent within sequence)
            label = sequence_df['Label'].iloc[0]
            
            # Extract features: 10 sensor features + 2 mouse coordinates = 12 features
            sensor_features = [
                'Channel_0_Filtered', 'Channel_0_Error',
                'Channel_1_Filtered', 'Channel_1_Error',
                'Channel_2_Filtered', 'Channel_2_Error',
                'Channel_3_Filtered', 'Channel_3_Error',
                'Channel_4_Filtered', 'Channel_4_Error'
            ]
            mouse_features = ['Mouse_X', 'Mouse_Y']
            
            feature_cols = sensor_features + mouse_features
            sequence_array = sequence_df[feature_cols].values.astype(np.float32)
            
            sequences.append((sequence_array, label))
        
        self.processing_stats['sequences_too_short'] += sequences_too_short
        self.processing_stats['sequences_extracted'] += len(sequences)
        
        if self.verbose:
            print(f"    Extracted {len(sequences)} sequences, {sequences_too_short} too short")
        
        return sequences
    
    def pad_or_truncate(self, sequence: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to fixed length."""
        seq_len = len(sequence)
        
        if seq_len >= self.max_sequence_length:
            # Truncate
            return sequence[:self.max_sequence_length]
        else:
            # Pad with zeros
            padding = np.zeros((self.max_sequence_length - seq_len, sequence.shape[1]), 
                             dtype=np.float32)
            return np.vstack([sequence, padding])
    
    def process_all_files(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Process all CSV files in data directory.
        
        Returns:
            X: Sequences array (n_samples, max_sequence_length, n_features)
            y: Encoded labels (n_samples,)
            label_names: List of label names
        """
        # Look for CSV files in multiple locations
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        # If no files found, try parent directory
        if len(csv_files) == 0:
            parent_dir = os.path.dirname(os.path.abspath(self.data_dir))
            csv_files = glob.glob(os.path.join(parent_dir, "*.csv"))
            if len(csv_files) > 0:
                print(f"No CSV files in {self.data_dir}, trying {parent_dir}...")
                self.data_dir = parent_dir
        
        all_sequences = []
        all_labels = []
        
        if self.verbose:
            print(f"Processing {len(csv_files)} CSV files from: {os.path.abspath(self.data_dir)}")
            print(f"Parameters: gap_threshold={self.gap_threshold_ms}ms, min_length={self.min_sequence_length}")
        
        for filepath in tqdm(csv_files, desc="📁 Processing CSV files", unit="file"):
            try:
                self.processing_stats['files_processed'] += 1
                df = self.load_csv_file(filepath)
                self.processing_stats['total_raw_samples'] += len(df)
                
                sequences = self.extract_sequences(df, filepath)
                
                for seq, label in sequences:
                    # Pad/truncate to fixed length
                    padded_seq = self.pad_or_truncate(seq)
                    all_sequences.append(padded_seq)
                    all_labels.append(label)
                    
            except Exception as e:
                self.processing_stats['files_failed'] += 1
                print(f"Error processing {filepath}: {e}")
                import traceback
                if self.verbose:
                    traceback.print_exc()
                continue
        
        # Check if we have any sequences
        if len(all_sequences) == 0:
            print("WARNING: No sequences extracted from CSV files!")
            print("This might be because:")
            print("  1. CSV files don't have enough data")
            print("  2. Time gaps are too large (increase gap_threshold_ms)")
            print("  3. Sequences are too short (decrease min_sequence_length)")
            # Return empty arrays with correct shape
            empty_X = np.array([]).reshape(0, self.max_sequence_length, 12)
            empty_y = np.array([], dtype=int)
            return empty_X, empty_y, []
        
        # Convert to numpy arrays
        X = np.array(all_sequences)
        
        # Encode labels
        if len(all_labels) > 0:
            y_encoded = self.label_encoder.fit_transform(all_labels)
            y = np.array(y_encoded)
            label_names = self.label_encoder.classes_.tolist()
        else:
            y = np.array([], dtype=int)
            label_names = []
            
        # Apply data quality cleaning if enabled
        cleaning_reports = []
        if False:  # DISABLED - self.enable_data_cleaning and len(X) > 0:
            if self.verbose:
                print(f"\nApplying data quality cleaning...")
            X, y, label_names, cleaning_reports = clean_gesture_data(X, y, label_names, self.verbose)
            self.cleaning_reports = cleaning_reports
        
        # Print processing statistics
        if self.verbose:
            print(f"\n=== Processing Summary ===")
            print(f"Files processed: {self.processing_stats['files_processed']}")
            print(f"Files failed: {self.processing_stats['files_failed']}")
            print(f"Total raw samples: {self.processing_stats['total_raw_samples']}")
            print(f"Large gaps found: {self.processing_stats['large_gaps_found']}")
            print(f"Sequences extracted: {self.processing_stats['sequences_extracted']}")
            print(f"Sequences too short: {self.processing_stats['sequences_too_short']}")
            print(f"Final sequences: {len(all_sequences)}")
            if len(all_sequences) > 0:
                print(f"Sequence shape: {X.shape}")
            print(f"Number of classes: {len(label_names)}")
            if label_names:
                print(f"Classes: {label_names}")
                
                # Class distribution
                from collections import Counter
                class_counts = Counter(all_labels)
                print(f"\n=== Class Distribution ===")
                # Sort class names as strings to avoid type conflicts
                sorted_classes = sorted(class_counts.keys(), key=str)
                for class_name in sorted_classes:
                    count = class_counts[class_name]
                    percentage = (count / len(all_labels)) * 100
                    print(f"{str(class_name):25}: {count:3d} sequences ({percentage:5.1f}%)")
        
        return X, y, label_names
    
    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced datasets."""
        from collections import Counter
        counts = Counter(y)
        total = len(y)
        
        weights = {}
        for class_id, count in counts.items():
            weights[class_id] = total / (len(counts) * count)
        
        return weights
    
    def extract_trajectory_coordinates(self, sequence: np.ndarray) -> np.ndarray:
        """
        Extract 2D trajectory coordinates from 5-plate sensor configuration.
        
        This is CRITICAL for universal gesture recognition - converts sensor readings
        to actual hand position coordinates for trajectory reconstruction training.
        
        Args:
            sequence: Shape (time_steps, 12) - raw sensor data
            
        Returns:
            coordinates: Shape (time_steps, 2) - normalized x,y trajectory coordinates
        """
        time_steps = sequence.shape[0]
        coordinates = np.zeros((time_steps, 2))
        
        for t in range(time_steps):
            # Extract filtered values for each plate (5-plate layout)
            # Plate positions: Left(0), Upper(1), Base(2), Right(3), Lower(4)
            left = sequence[t, 0]      # Channel_0_Filtered  
            upper = sequence[t, 2]     # Channel_1_Filtered
            base = sequence[t, 4]      # Channel_2_Filtered (reference/baseline)
            right = sequence[t, 6]     # Channel_3_Filtered
            lower = sequence[t, 8]     # Channel_4_Filtered
            
            # Method 1: Differential positioning (most accurate for gestures)
            # X-coordinate: Left vs Right plate activation difference
            left_right_sum = abs(left) + abs(right) + 1e-8  # Avoid division by zero
            x_differential = (right - left) / left_right_sum
            
            # Y-coordinate: Upper vs Lower plate activation difference  
            upper_lower_sum = abs(upper) + abs(lower) + 1e-8
            y_differential = (upper - lower) / upper_lower_sum
            
            # Method 2: Weighted centroid (accounts for base plate influence)
            base_influence = max(abs(base), 0.1)  # Minimum baseline influence
            
            # Weight coordinates by base plate activation (stronger signal = more weight)
            x_weighted = x_differential * (base_influence / (base_influence + 1.0))
            y_weighted = y_differential * (base_influence / (base_influence + 1.0))
            
            # Method 3: Spatial smoothing (reduce sensor noise)
            if t > 0:
                # Apply temporal smoothing with previous coordinate
                smoothing_factor = 0.3
                x_smoothed = smoothing_factor * coordinates[t-1, 0] + (1-smoothing_factor) * x_weighted
                y_smoothed = smoothing_factor * coordinates[t-1, 1] + (1-smoothing_factor) * y_weighted
            else:
                x_smoothed, y_smoothed = x_weighted, y_weighted
            
            # Normalize to [-1, 1] range with saturation
            coordinates[t, 0] = np.tanh(x_smoothed * 2.0)  # X coordinate (Left=-1, Right=+1)
            coordinates[t, 1] = np.tanh(y_smoothed * 2.0)  # Y coordinate (Lower=-1, Upper=+1)
        
        return coordinates
    
    def process_all_files_with_trajectories(self) -> tuple:
        """
        Process all CSV files and extract BOTH sensor features AND trajectory coordinates.
        
        This is the enhanced version for universal gesture recognition that provides
        both discrete classification labels AND continuous trajectory coordinates.
        
        Returns:
            Tuple of (X_sensors, X_trajectories, y_labels, label_names, stats)
        """
        csv_files = []
        for pattern in ['*.csv', '*.CSV']:
            csv_files.extend(glob.glob(os.path.join(self.data_dir, pattern)))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return np.array([]), np.array([]), np.array([]), [], {}
        
        all_sensor_sequences = []
        all_trajectory_sequences = []
        all_labels = []
        
        if self.verbose:
            print(f"Processing {len(csv_files)} CSV files from: {self.data_dir}")
            print(f"Parameters: gap_threshold={self.gap_threshold_ms}ms, min_length={self.min_sequence_length}")
        
        # Process each CSV file with progress bar
        for filepath in tqdm(csv_files, desc="📁 Processing CSV files", unit="file"):
            try:
                self.processing_stats['files_processed'] += 1
                df = self.load_csv_file(filepath)
                self.processing_stats['total_raw_samples'] += len(df)
                
                sequences = self.extract_sequences(df, filepath)
                
                for seq, label in sequences:
                    # Pad/truncate to fixed length
                    padded_seq = self.pad_or_truncate(seq)
                    
                    # Extract trajectory coordinates from sensor data
                    trajectory_coords = self.extract_trajectory_coordinates(padded_seq)
                    
                    all_sensor_sequences.append(padded_seq)
                    all_trajectory_sequences.append(trajectory_coords)
                    all_labels.append(label)
                    
            except Exception as e:
                self.processing_stats['files_failed'] += 1
                print(f"Error processing {filepath}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        if len(all_sensor_sequences) == 0:
            print("No sequences extracted from any files!")
            return np.array([]), np.array([]), np.array([]), [], self.processing_stats
        
        # Convert to numpy arrays
        X_sensors = np.array(all_sensor_sequences)
        X_trajectories = np.array(all_trajectory_sequences) 
        
        # Encode labels
        self.label_encoder.fit(all_labels)
        y = self.label_encoder.transform(all_labels)
        label_names = self.label_encoder.classes_.tolist()
        
        # Apply data quality cleaning if enabled (DISABLED - found to corrupt data)
        cleaning_reports = []
        if False:  # DISABLED - self.enable_data_cleaning and len(X_sensors) > 0:
            if self.verbose:
                print(f"\nApplying data quality cleaning...")
            X_sensors, y, label_names, cleaning_reports = clean_gesture_data(X_sensors, y, label_names, self.verbose)
            self.cleaning_reports = cleaning_reports
        
        # Print processing statistics
        if self.verbose:
            print(f"\n=== Processing Summary ===")
            print(f"Files processed: {self.processing_stats['files_processed']}")
            print(f"Files failed: {self.processing_stats['files_failed']}")
            print(f"Total raw samples: {self.processing_stats['total_raw_samples']}")
            print(f"Large gaps found: {self.processing_stats['large_gaps_found']}")
            print(f"Sequences extracted: {self.processing_stats['sequences_extracted']}")
            print(f"Sequences too short: {self.processing_stats['sequences_too_short']}")
            print(f"Final sequences: {len(X_sensors)}")
            print(f"Sensor data shape: {X_sensors.shape}")
            print(f"Trajectory data shape: {X_trajectories.shape}")
            print(f"Number of classes: {len(label_names)}")
            print(f"Classes: {label_names}")
            
            print(f"\n=== Class Distribution ===")
            class_counts = {}
            for class_id in y:
                class_name = label_names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name in sorted(class_counts.keys()):
                count = class_counts[class_name]
                percentage = (count / len(y)) * 100
                print(f"{class_name:25s}: {count:3d} sequences ({percentage:5.1f}%)")
        
        return X_sensors, X_trajectories, y, label_names, self.processing_stats

