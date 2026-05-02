#!/usr/bin/env python3
"""
Complete Trajectory Reconstruction Training Pipeline

Automated pipeline that:
1. Auto-detects CSV files in specified directory
2. Processes sensor data and extracts trajectory coordinates  
3. Applies feature engineering and data augmentation
4. Trains trajectory-only model with CPU optimization
5. Saves model and generates evaluation metrics

Optimized for limited data (70 classes × 30 samples) with 10x augmentation.
"""

import os
import sys

# Fix GLIBCXX compatibility issue - use conda's libstdc++ if available
# This must be done BEFORE importing tensorflow/pandas
conda_env = os.environ.get('CONDA_PREFIX')
conda_env_name = os.environ.get('CONDA_DEFAULT_ENV', '')
user_home = os.environ.get('HOME', '')

# Try to find conda lib directory
conda_lib_paths = []

# Check CONDA_PREFIX first
if conda_env:
    conda_lib_paths.append(os.path.join(conda_env, 'lib'))

# Check common conda locations
if user_home:
    conda_lib_paths.extend([
        os.path.join(user_home, 'anaconda3', 'envs', conda_env_name, 'lib'),
        os.path.join(user_home, 'miniconda3', 'envs', conda_env_name, 'lib'),
    ])

# Check specific VM paths
conda_lib_paths.extend([
    '/DATA/sarmistha_2221cs21/anaconda3/envs/rishu39/lib',
    '/DATA/sarmistha_2221cs21/anaconda3/envs/rishu/lib',
    '/DATA/sarmistha_2221cs21/anaconda3/lib',
])

# Add conda lib to LD_LIBRARY_PATH if it exists
for lib_path in conda_lib_paths:
    if lib_path and os.path.exists(lib_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if lib_path not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{current_ld_path}"
            # Also set library path for ctypes
            try:
                import ctypes
                ctypes.CDLL(os.path.join(lib_path, 'libstdc++.so.6'))
            except:
                pass
        break

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings

# GPU/CPU configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable GPU if available (auto-detect)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Set memory limit to prevent OOM errors (optional - can be adjusted)
            # tf.config.experimental.set_memory_growth(gpu, False)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            # )
        
        print(f"✅ GPU detected: {len(gpus)} GPU(s) available")
        print(f"   Using GPU: {gpus[0].name}")
        print(f"   💡 Memory growth enabled to prevent OOM errors")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
        print("   Falling back to CPU mode")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    print("⚠️ No GPU detected, using CPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from data_processing import GestureDataProcessor
from feature_engineering import RobustFeatureEngineer
from augmentation import ClassBalancingAugmentation
from models.trajectory_only_model import TrajectoryOnlyModel
import glob


class TrajectoryAugmentation:
    """Specialized augmentation that maintains sensor-trajectory correspondence."""
    
    def __init__(self, noise_std: float = 0.02, time_warp_range: Tuple[float, float] = (0.8, 1.2)):
        """Initialize trajectory-aware augmentation."""
        self.noise_std = noise_std
        self.time_warp_range = time_warp_range
    
    def augment_trajectory_pair(self, sensor_features: np.ndarray, 
                               trajectory_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment sensor features and trajectory coordinates while maintaining correspondence.
        
        Args:
            sensor_features: Shape (seq_len, n_features) - sensor data
            trajectory_coords: Shape (seq_len, 2) - trajectory coordinates
            
        Returns:
            Tuple of (augmented_sensors, augmented_trajectory)
        """
        # Apply the same random transformations to both
        
        # 1. Add correlated noise
        signal_range_sensor = np.max(sensor_features) - np.min(sensor_features)
        signal_range_traj = np.max(trajectory_coords) - np.min(trajectory_coords)
        
        noise_sensor = np.random.normal(0, self.noise_std * signal_range_sensor, sensor_features.shape)
        noise_traj = np.random.normal(0, self.noise_std * signal_range_traj * 0.5, trajectory_coords.shape)  # Less noise on trajectory
        
        augmented_sensors = sensor_features + noise_sensor
        augmented_trajectory = trajectory_coords + noise_traj
        
        # 2. Time warping (same warp factor for both)
        warp_factor = np.random.uniform(*self.time_warp_range)
        
        if warp_factor != 1.0:
            seq_len = len(sensor_features)
            orig_indices = np.arange(seq_len)
            new_indices = np.linspace(0, seq_len - 1, int(seq_len * warp_factor))
            
            # Interpolate sensor features
            from scipy.interpolate import interp1d
            warped_sensors = np.zeros_like(sensor_features)
            for i in range(sensor_features.shape[1]):
                interp_func = interp1d(orig_indices, augmented_sensors[:, i], 
                                     kind='linear', fill_value='extrapolate')
                warped_values = interp_func(new_indices)
                
                # Resample to original length
                if len(warped_values) != seq_len:
                    resample_indices = np.linspace(0, len(warped_values) - 1, seq_len)
                    interp_func2 = interp1d(np.arange(len(warped_values)), warped_values,
                                          kind='linear', fill_value='extrapolate')
                    warped_sensors[:, i] = interp_func2(resample_indices)
                else:
                    warped_sensors[:, i] = warped_values
            
            # Interpolate trajectory coordinates  
            warped_trajectory = np.zeros_like(trajectory_coords)
            for i in range(trajectory_coords.shape[1]):
                interp_func = interp1d(orig_indices, augmented_trajectory[:, i],
                                     kind='linear', fill_value='extrapolate')
                warped_values = interp_func(new_indices)
                
                # Resample to original length
                if len(warped_values) != seq_len:
                    resample_indices = np.linspace(0, len(warped_values) - 1, seq_len)
                    interp_func2 = interp1d(np.arange(len(warped_values)), warped_values,
                                          kind='linear', fill_value='extrapolate')
                    warped_trajectory[:, i] = interp_func2(resample_indices)
                else:
                    warped_trajectory[:, i] = warped_values
            
            augmented_sensors = warped_sensors
            augmented_trajectory = warped_trajectory
        
        return augmented_sensors, augmented_trajectory


class TrajectoryTrainingPipeline:
    """Complete training pipeline for trajectory reconstruction."""
    
    def __init__(self, 
                 data_dir: str = ".",
                 output_dir: str = "./trajectory_model",
                 verbose: bool = True):
        """
        Initialize trajectory training pipeline.
        
        Args:
            data_dir: Directory containing CSV files
            output_dir: Output directory for model and results
            verbose: Enable detailed logging
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Data storage
        self.X_sensors = None
        self.X_trajectories = None
        self.y_labels = None
        self.label_names = None
        
        # Model and training
        self.model = None
        self.history = None
        self.scalers = {}
        
        # Statistics
        self.training_stats = {}
    
    def discover_csv_files(self) -> List[Path]:
        """Automatically discover CSV files in the data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        print(f"📁 Discovered {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"   - {csv_file.name}")
        
        return csv_files
    
    def load_and_process_data(self,
                             min_sequence_length: int = 20,
                             max_sequence_length: int = 500,
                             gap_threshold_ms: float = 2000.0) -> Dict:
        """
        Load and process all CSV files with trajectory extraction.
        
        Args:
            min_sequence_length: Minimum sequence length to keep
            max_sequence_length: Maximum sequence length (pad/truncate)
            gap_threshold_ms: Time gap threshold for sequence splitting
            
        Returns:
            Processing statistics
        """
        print("=" * 80)
        print("🔄 STEP 1: DATA LOADING & TRAJECTORY EXTRACTION")
        print("=" * 80)
        
        # Discover CSV files
        csv_files = self.discover_csv_files()
        
        # Initialize data processor
        processor = GestureDataProcessor(
            data_dir=str(self.data_dir),
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            gap_threshold_ms=gap_threshold_ms,
            enable_data_cleaning=False,  # Disable to prevent data corruption
            verbose=self.verbose
        )
        
        # Process files with trajectory extraction
        print("📊 Processing CSV files and extracting trajectories...")
        
        # Use existing method with trajectory extraction
        X_sensors, X_trajectories, y_labels, label_names, processing_stats = processor.process_all_files_with_trajectories()
        
        if len(X_sensors) == 0:
            raise ValueError("No valid sequences extracted from CSV files")
        
        # Store sensor features and trajectory coordinates
        self.X_sensors = X_sensors  # Shape: (n_samples, seq_len, 12)
        self.X_trajectories = X_trajectories  # Shape: (n_samples, seq_len, 2)
        self.y_labels = y_labels
        self.label_names = label_names
        
        # Statistics
        stats = {
            'total_sequences': len(self.X_sensors),
            'sequence_length': self.X_sensors.shape[1],
            'sensor_features': self.X_sensors.shape[2],
            'coordinate_dim': self.X_trajectories.shape[2],
            'unique_classes': len(np.unique(self.y_labels)),
            'class_names': label_names
        }
        
        print(f"✅ Data loaded successfully:")
        print(f"   📊 Total sequences: {stats['total_sequences']}")
        print(f"   📏 Sequence length: {stats['sequence_length']}")
        print(f"   🔧 Sensor features: {stats['sensor_features']}")
        print(f"   🎯 Trajectory dimensions: {stats['coordinate_dim']}")
        print(f"   📂 Classes: {stats['unique_classes']}")
        
        return stats
    
    def apply_feature_engineering(self) -> Dict:
        """Apply advanced feature engineering to sensor data."""
        print("=" * 80)
        print("🛠️ STEP 2: FEATURE ENGINEERING")
        print("=" * 80)
        
        # Initialize robust feature engineer
        feature_engineer = RobustFeatureEngineer(
            enable_spatial_features=True,
            use_mouse_coords=False,  # Don't use mouse coords for trajectory-only model
            normalize_mouse=False,
            outlier_detection=True
        )
        
        # Process sensor features
        print("🔧 Applying feature engineering to sensor data...")
        X_engineered, _ = feature_engineer.process_features(
            self.X_sensors, 
            self.y_labels, 
            fit=True
        )
        
        # Update sensor data
        self.X_sensors = X_engineered
        self.scalers['feature_engineer'] = feature_engineer
        
        print(f"✅ Feature engineering complete:")
        print(f"   📊 Enhanced features: {X_engineered.shape[2]}")
        print(f"   📏 Sequence length: {X_engineered.shape[1]}")
        
        return {
            'original_features': 12,
            'engineered_features': X_engineered.shape[2],
            'feature_expansion_ratio': X_engineered.shape[2] / 12
        }
    
    def apply_augmentation(self, target_samples_per_class: int = 30) -> Dict:
        """Apply trajectory-aware data augmentation."""
        print("=" * 80)
        print("🔄 STEP 3: TRAJECTORY-AWARE AUGMENTATION") 
        print("=" * 80)
        
        # Calculate current class distribution
        unique_classes, class_counts = np.unique(self.y_labels, return_counts=True)
        print(f"📊 Original class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            class_name = self.label_names[cls] if cls < len(self.label_names) else f"Class_{cls}"
            print(f"   {class_name}: {count} samples")
        
        # Initialize trajectory augmentation
        traj_augmenter = TrajectoryAugmentation()
        
        # Calculate augmentation needed per class
        augmented_sensors = []
        augmented_trajectories = []
        augmented_labels = []
        
        for cls in tqdm(unique_classes, desc="Augmenting classes"):
            # Get samples for this class
            class_mask = self.y_labels == cls
            class_sensors = self.X_sensors[class_mask]
            class_trajectories = self.X_trajectories[class_mask]
            
            current_count = len(class_sensors)
            augmentation_needed = max(0, target_samples_per_class - current_count)
            
            # Add original samples
            augmented_sensors.append(class_sensors)
            augmented_trajectories.append(class_trajectories)
            augmented_labels.extend([cls] * current_count)
            
            # Generate augmented samples
            if augmentation_needed > 0:
                for _ in range(augmentation_needed):
                    # Randomly select a sample to augment
                    idx = np.random.randint(0, current_count)
                    sensor_sample = class_sensors[idx]
                    trajectory_sample = class_trajectories[idx]
                    
                    # Apply augmentation
                    aug_sensor, aug_trajectory = traj_augmenter.augment_trajectory_pair(
                        sensor_sample, trajectory_sample
                    )
                    
                    augmented_sensors.append(aug_sensor[np.newaxis, :, :])
                    augmented_trajectories.append(aug_trajectory[np.newaxis, :, :])
                    augmented_labels.append(cls)
        
        # Concatenate all augmented data
        self.X_sensors = np.concatenate(augmented_sensors, axis=0)
        self.X_trajectories = np.concatenate(augmented_trajectories, axis=0)
        self.y_labels = np.array(augmented_labels)
        
        # Statistics
        final_unique, final_counts = np.unique(self.y_labels, return_counts=True)
        print(f"✅ Augmentation complete:")
        print(f"   📊 Total samples: {len(self.y_labels)}")
        print(f"   🎯 Target per class: {target_samples_per_class}")
        print(f"   📈 Augmentation ratio: {len(self.y_labels) / len(unique_classes) / np.mean(class_counts):.1f}x")
        
        return {
            'original_samples': int(len(class_counts) * np.mean(class_counts)),
            'augmented_samples': int(len(self.y_labels)),
            'augmentation_ratio': float(len(self.y_labels) / (len(class_counts) * np.mean(class_counts))),
            'samples_per_class': {int(k): int(v) for k, v in zip(final_unique, final_counts)}
        }
    
    def prepare_training_data(self, validation_split: float = 0.2) -> Dict:
        """Prepare and split data for training."""
        print("=" * 80)
        print("📊 STEP 4: TRAINING DATA PREPARATION")
        print("=" * 80)
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Normalize trajectory coordinates per-trajectory (preserves patterns)
        # This is CRITICAL: normalizing globally destroys trajectory structure
        n_samples, seq_len, coord_dim = self.X_trajectories.shape
        
        print("🔧 Normalizing trajectories per-sample (preserves trajectory patterns)...")
        for i in tqdm(range(n_samples), desc="Normalizing trajectories"):
            traj = self.X_trajectories[i].copy()
            
            # Normalize each trajectory independently to [-1, 1] range
            # This preserves the relative shape and direction of each trajectory
            traj_min = np.min(traj, axis=0, keepdims=True)
            traj_max = np.max(traj, axis=0, keepdims=True)
            traj_range = traj_max - traj_min + 1e-8  # Avoid division by zero
            
            # Normalize to [0, 1] then scale to [-1, 1]
            normalized = 2.0 * (traj - traj_min) / traj_range - 1.0
            self.X_trajectories[i] = normalized
            
            # Store normalization params for each trajectory (for inverse transform)
            if 'trajectory_norms' not in self.scalers:
                self.scalers['trajectory_norms'] = []
            self.scalers['trajectory_norms'].append({
                'min': traj_min.flatten(),
                'max': traj_max.flatten(),
                'range': traj_range.flatten()
            })
        
        print(f"✅ Trajectory normalization complete (per-sample normalization)")
        
        # Split data
        X_train_sensors, X_val_sensors, X_train_traj, X_val_traj, y_train, y_val = train_test_split(
            self.X_sensors, 
            self.X_trajectories, 
            self.y_labels,
            test_size=validation_split,
            stratify=self.y_labels,
            random_state=42
        )
        
        print(f"✅ Training data prepared:")
        print(f"   🏋️ Training samples: {len(X_train_sensors)}")
        print(f"   🧪 Validation samples: {len(X_val_sensors)}")
        print(f"   📊 Sensor features shape: {X_train_sensors.shape}")
        print(f"   🎯 Trajectory shape: {X_train_traj.shape}")
        
        return {
            'X_train_sensors': X_train_sensors,
            'X_val_sensors': X_val_sensors,
            'X_train_trajectories': X_train_traj,
            'X_val_trajectories': X_val_traj,
            'y_train': y_train,
            'y_val': y_val,
            'train_samples': len(X_train_sensors),
            'val_samples': len(X_val_sensors)
        }
    
    def train_model(self, 
                   training_data: Dict,
                   epochs: int = 100,
                   batch_size: int = 12,
                   patience: int = 15) -> Dict:
        """Train the trajectory-only model."""
        print("=" * 80)
        print("🚀 STEP 5: MODEL TRAINING")
        print("=" * 80)
        
        # Extract training data
        X_train = training_data['X_train_sensors']
        X_val = training_data['X_val_sensors']
        y_train = training_data['X_train_trajectories'] 
        y_val = training_data['X_val_trajectories']
        
        # Create model with reduced size and better regularization
        self.model = TrajectoryOnlyModel(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2],
            coordinate_dim=2,
            lstm_units=32,  # Reduced from default 64
            cnn_filters=(32, 16, 8),  # Reduced from (64, 32, 16)
            dropout_rate=0.5,  # Increased from 0.3
            learning_rate=0.0005  # Lower initial learning rate
        )
        
        model = self.model.build()
        self.model.summary()
        
        # Custom callback for epoch progress tracking with tqdm
        class EpochProgressCallback(keras.callbacks.Callback):
            def __init__(self, total_epochs):
                super().__init__()
                self.total_epochs = total_epochs
                self.pbar = None
                self.current_epoch = 0
                self.early_stopped = False
                
            def on_train_begin(self, logs=None):
                self.pbar = tqdm(total=self.total_epochs, desc='Training Progress', 
                                unit='epoch', ncols=100, 
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                
            def on_epoch_end(self, epoch, logs=None):
                self.current_epoch = epoch + 1
                # Update progress bar with metrics
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                mae = logs.get('mae', 0)
                val_mae = logs.get('val_mae', 0)
                
                self.pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'val_loss': f'{val_loss:.4f}',
                    'mae': f'{mae:.4f}',
                    'val_mae': f'{val_mae:.4f}'
                })
                self.pbar.update(1)
                
            def on_train_end(self, logs=None):
                if self.pbar:
                    if self.current_epoch < self.total_epochs:
                        # Early stopping occurred
                        self.pbar.set_postfix({'status': 'Early stopped'})
                    self.pbar.close()
                    print()  # New line after progress bar
        
        # Setup callbacks
        callbacks = [
            # Custom epoch progress callback
            EpochProgressCallback(total_epochs=epochs),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        print(f"🏋️ Starting training:")
        print(f"   📊 Training samples: {len(X_train)}")
        print(f"   🧪 Validation samples: {len(X_val)}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Max epochs: {epochs}")
        print(f"   ⏰ Early stopping patience: {patience}")
        print()
        
        # Train model with tqdm progress tracking
        # Try training with error handling for OOM
        try:
            self.history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0  # Use custom callback instead
            )
        except tf.errors.ResourceExhaustedError as e:
            print(f"\n❌ GPU Out of Memory (OOM) error with batch_size={batch_size}")
            print(f"   💡 Reducing batch size to {batch_size // 2} and retrying...")
            print(f"   💡 You can also set --batch_size 8 or lower for more stability\n")
            
            # Clear GPU memory
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
            # Retry with smaller batch size
            reduced_batch_size = max(4, batch_size // 2)
            self.history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=reduced_batch_size,
                callbacks=callbacks,
                verbose=0
            )
            print(f"✅ Training successful with reduced batch_size={reduced_batch_size}")
        
        # Training statistics
        final_epoch = len(self.history.history['loss'])
        best_val_loss = min(self.history.history['val_loss'])
        best_train_loss = min(self.history.history['loss'])
        
        print(f"✅ Training completed:")
        print(f"   🎯 Final epoch: {final_epoch}")
        print(f"   📉 Best validation loss: {best_val_loss:.6f}")
        print(f"   📊 Best training loss: {best_train_loss:.6f}")
        
        return {
            'final_epoch': final_epoch,
            'best_val_loss': best_val_loss,
            'best_train_loss': best_train_loss,
            'training_history': self.history.history
        }
    
    def evaluate_and_save_model(self, training_data: Dict) -> Dict:
        """Evaluate model performance and save results."""
        print("=" * 80)
        print("📊 STEP 6: MODEL EVALUATION & SAVING")
        print("=" * 80)
        
        # Evaluate on validation data
        X_val = training_data['X_val_sensors']
        y_val = training_data['X_val_trajectories']
        
        val_loss, val_mae, val_mse = self.model.model.evaluate(X_val, y_val, verbose=0)
        
        # Generate predictions for analysis
        y_pred = self.model.model.predict(X_val, verbose=0)
        
        # Calculate trajectory-specific metrics
        trajectory_errors = []
        correlation_scores = []
        
        for i in range(len(y_val)):
            pred_traj = y_pred[i]
            true_traj = y_val[i]
            
            # Trajectory-level MSE
            traj_mse = np.mean((pred_traj - true_traj)**2)
            trajectory_errors.append(traj_mse)
            
            # Correlation between predicted and true trajectories
            pred_flat = pred_traj.flatten()
            true_flat = true_traj.flatten()
            if np.std(pred_flat) > 0 and np.std(true_flat) > 0:
                corr = np.corrcoef(pred_flat, true_flat)[0, 1]
                correlation_scores.append(corr if not np.isnan(corr) else 0.0)
            else:
                correlation_scores.append(0.0)
        
        # Save model with metadata
        model_path = self.output_dir / 'trajectory_model.h5'
        self.model.save_model(str(model_path), save_metadata=True)
        
        # Save scalers
        import joblib
        scaler_path = self.output_dir / 'scalers.pkl'
        joblib.dump(self.scalers, scaler_path)
        print(f"✅ Scalers saved to: {scaler_path}")
        
        # Generate evaluation plots
        self._generate_evaluation_plots(training_data, y_pred)
        
        # Evaluation results
        eval_results = {
            'validation_loss': float(val_loss),
            'validation_mae': float(val_mae),
            'validation_mse': float(val_mse),
            'mean_trajectory_error': float(np.mean(trajectory_errors)),
            'std_trajectory_error': float(np.std(trajectory_errors)),
            'mean_correlation': float(np.mean(correlation_scores)),
            'std_correlation': float(np.std(correlation_scores)),
            'trajectory_accuracy_percentage': float(np.mean(np.array(correlation_scores) > 0.5) * 100)
        }
        
        print(f"📊 Final Model Performance:")
        print(f"   🎯 Validation Loss (MSE): {eval_results['validation_loss']:.6f}")
        print(f"   📏 Validation MAE: {eval_results['validation_mae']:.6f}")
        print(f"   📊 Mean Trajectory Error: {eval_results['mean_trajectory_error']:.6f}")
        print(f"   🔗 Mean Correlation: {eval_results['mean_correlation']:.3f}")
        print(f"   ✅ Trajectory Accuracy (>50% corr): {eval_results['trajectory_accuracy_percentage']:.1f}%")
        
        # Save evaluation results
        eval_path = self.output_dir / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        return eval_results
    
    def _generate_evaluation_plots(self, training_data: Dict, y_pred: np.ndarray):
        """Generate evaluation plots and save them."""
        plt.style.use('default')
        
        # 1. Training history plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss', alpha=0.8)
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE curves
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE', alpha=0.8)
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE', alpha=0.8)
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sample trajectory comparison
        X_val = training_data['X_val_sensors']
        y_val = training_data['X_val_trajectories']
        
        # Plot first few trajectory predictions
        for i in range(min(3, len(y_val))):
            pred_traj = y_pred[i]
            true_traj = y_val[i]
            
            # X coordinates
            axes[1, 0].plot(true_traj[:, 0], label=f'True X_{i}', alpha=0.7, linestyle='-')
            axes[1, 0].plot(pred_traj[:, 0], label=f'Pred X_{i}', alpha=0.7, linestyle='--')
        
        axes[1, 0].set_title('X Coordinate Predictions')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('X Coordinate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Y coordinates
        for i in range(min(3, len(y_val))):
            pred_traj = y_pred[i]
            true_traj = y_val[i]
            
            axes[1, 1].plot(true_traj[:, 1], label=f'True Y_{i}', alpha=0.7, linestyle='-')
            axes[1, 1].plot(pred_traj[:, 1], label=f'Pred Y_{i}', alpha=0.7, linestyle='--')
        
        axes[1, 1].set_title('Y Coordinate Predictions')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Y Coordinate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'training_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 2D Trajectory visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i in range(min(6, len(y_val))):
            row, col = i // 3, i % 3
            pred_traj = y_pred[i]
            true_traj = y_val[i]
            
            axes[row, col].plot(true_traj[:, 0], true_traj[:, 1], 'b-', 
                               label='True Trajectory', linewidth=2, alpha=0.8)
            axes[row, col].plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', 
                               label='Predicted Trajectory', linewidth=2, alpha=0.8)
            
            # Mark start and end points
            axes[row, col].scatter(true_traj[0, 0], true_traj[0, 1], 
                                  color='green', s=100, marker='o', label='Start', zorder=5)
            axes[row, col].scatter(true_traj[-1, 0], true_traj[-1, 1], 
                                  color='red', s=100, marker='x', label='End', zorder=5)
            
            axes[row, col].set_title(f'Trajectory Sample {i+1}')
            axes[row, col].set_xlabel('X Coordinate')
            axes[row, col].set_ylabel('Y Coordinate')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'trajectory_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Evaluation plots saved to: {self.output_dir / 'plots'}")
    
    def run_complete_pipeline(self,
                            min_sequence_length: int = 20,
                            max_sequence_length: int = 500,
                            target_samples_per_class: int = 30,
                            epochs: int = 100,
                            batch_size: int = 12,
                            patience: int = 15) -> Dict:
        """Run the complete trajectory training pipeline."""
        start_time = datetime.now()
        
        print("🚀" * 40)
        print("🚀 COMPLETE TRAJECTORY RECONSTRUCTION PIPELINE")
        print("🚀" * 40)
        
        try:
            # Step 1: Load and process data
            data_stats = self.load_and_process_data(
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length
            )
            
            # Step 2: Feature engineering
            feature_stats = self.apply_feature_engineering()
            
            # Step 3: Data augmentation
            aug_stats = self.apply_augmentation(target_samples_per_class=target_samples_per_class)
            
            # Step 4: Prepare training data
            training_data = self.prepare_training_data()
            
            # Step 5: Train model
            training_stats = self.train_model(
                training_data=training_data,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            # Step 6: Evaluate and save
            eval_results = self.evaluate_and_save_model(training_data)
            
            # Pipeline summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            pipeline_results = {
                'pipeline_duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'data_statistics': data_stats,
                'feature_statistics': feature_stats,
                'augmentation_statistics': aug_stats,
                'training_statistics': training_stats,
                'evaluation_results': eval_results,
                'model_path': str(self.output_dir / 'trajectory_model.h5'),
                'scalers_path': str(self.output_dir / 'scalers.pkl')
            }
            
            # Save complete results
            results_path = self.output_dir / 'pipeline_results.json'
            with open(results_path, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            print("🎉" * 40)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("🎉" * 40)
            print(f"⏰ Total duration: {duration/60:.1f} minutes")
            print(f"🎯 Final trajectory accuracy: {eval_results['trajectory_accuracy_percentage']:.1f}%")
            print(f"📁 Model saved to: {self.output_dir}")
            print("🎉" * 40)
            
            return pipeline_results
            
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Train Trajectory Reconstruction Model')
    parser.add_argument('--data_dir', type=str, default='.', 
                       help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='./trajectory_model',
                       help='Output directory for model and results')
    parser.add_argument('--min_seq_len', type=int, default=20,
                       help='Minimum sequence length')
    parser.add_argument('--max_seq_len', type=int, default=500,
                       help='Maximum sequence length')
    parser.add_argument('--target_samples', type=int, default=30,
                       help='Target samples per class after augmentation')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=12,
                       help='Training batch size')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = TrajectoryTrainingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    results = pipeline.run_complete_pipeline(
        min_sequence_length=args.min_seq_len,
        max_sequence_length=args.max_seq_len,
        target_samples_per_class=args.target_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience
    )
    
    return results


if __name__ == "__main__":
    main()
