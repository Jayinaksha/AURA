# AURA — Touchless Gesture-Based Cursor Control

AURA is a touchless trackpad system that lets you control your cursor by moving your hand or finger over a capacitive sensor array — no physical contact needed. It uses an **MPR121 capacitive sensor** connected to an Arduino to read 5-channel electromagnetic proximity data, which is fed into a machine learning pipeline to predict (x, y) cursor position in real time.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [1. Flash the Arduino Firmware](#1-flash-the-arduino-firmware)
  - [2. Install Python Dependencies](#2-install-python-dependencies)
  - [3. Collect Training Data](#3-collect-training-data)
  - [4. Train the Model](#4-train-the-model)
  - [5. Run the Live Demo](#5-run-the-live-demo)
- [Pipeline Overview](#pipeline-overview)
- [Configuration](#configuration)
- [Models](#models)
- [File Reference](#file-reference)
- [License](#license)

---

## How It Works

1. **Sensor**: An MPR121 capacitive sensor with 5 electrodes is connected to an Arduino. It measures the electromagnetic field disturbance caused by a hand moving above it.
2. **Serial Output**: The Arduino streams filtered capacitance values and error signals (baseline – filtered) for each of the 5 channels over USB serial at 2 Mbps.
3. **Data Collection**: Python scripts capture this serial data alongside cursor coordinates (either from a moving mouse pattern or from a static grid) to build a labeled training dataset.
4. **ML Model**: A machine learning model (Random Forest or deep learning CNN-BiLSTM) is trained to map the 4 most informative error channels to screen (x, y) coordinates.
5. **Real-Time Inference**: The trained model runs on live serial data, predicts cursor position, applies smoothing (EMA or Kalman filter), and drives a fullscreen cursor overlay.

---

## Hardware Requirements

| Component | Details |
|---|---|
| Microcontroller | Arduino (tested on boards with `/dev/ttyACM0`) |
| Capacitive Sensor | NXP MPR121 (5 channels used) |
| Interface | I²C |
| USB | Standard USB-B/micro for serial communication |

---

## Software Requirements

- Python 3.8+
- Arduino IDE (to flash the firmware)
- Arduino libraries: [MPR121](https://github.com/janelia-arduino/MPR121), Streaming

### Python Packages

```bash
pip install pyserial numpy scikit-learn joblib tensorflow pandas pyautogui
```

For the deep learning trajectory model (`emt_live.py`):

```bash
pip install tensorflow keras
```

---

## Project Structure

```
AURA/
├── Test_emt_copy/              # Arduino firmware
│   ├── Test_emt_copy.ino       # Main firmware sketch
│   ├── Constants.h             # Sensor configuration constants (header)
│   └── Constants.cpp           # Sensor configuration constants (implementation)
│
├── data/                       # Collected training datasets (CSV files)
│   ├── mpr121_error_data*.csv  # Dynamic trajectory recordings
│   ├── mpr121_static_grid_data*.csv  # Static grid recordings
│   └── mpr121_user_pattern_data.csv  # Pattern-following recordings
│
├── Archive/                    # Research/training scripts
│   ├── data_processing.py      # Sequence extraction and label assignment
│   ├── feature_engineering.py  # Feature construction for ML models
│   ├── augmentation.py         # Data augmentation utilities
│   ├── train_trajectory_model.py # Full training pipeline (CNN-BiLSTM)
│   └── emt_live.py             # Archived version of live inference
│
├── py_mat.py                   # Basic serial-to-CSV data logger
├── data_rec.py                 # Pattern-following data recorder (circle/line)
├── rec_grid.py                 # Static grid data recorder (15×15 grid)
├── aura_demo.py                # Live demo — Random Forest + EMA smoothing
├── inference.py                # Live demo — cursor_model + Kalman filter
├── emt_live.py                 # Live demo — CNN-BiLSTM trajectory model
│
├── rf_model.pkl                # Trained Random Forest model
├── scaler_X.pkl                # Feature scaler (StandardScaler) for RF model
├── cursor_model.pkl            # Trained cursor model dict (with normalization stats)
├── trajectory_model.h5         # Trained CNN-BiLSTM trajectory model
├── error1_graph.png            # Training error graph
├── error2_graph.png            # Training error graph
└── LICENSE
```

---

## Getting Started

### 1. Flash the Arduino Firmware

1. Open `Test_emt_copy/Test_emt_copy.ino` in the Arduino IDE.
2. Install the **MPR121** and **Streaming** libraries via the Library Manager.
3. Connect your Arduino with the MPR121 sensor on I²C.
4. Select your board and port, then upload the sketch.
5. Open the Serial Monitor at **2,000,000 bps** to verify output like:

   ```
    Channel 0: 967, error :-4 Channel 1: 820, error :12 ...
   ```

> **Calibration**: The firmware automatically calibrates the baseline on startup. Keep your hand away from the sensor for the first few seconds after power-on.

---

### 2. Install Python Dependencies

```bash
pip install pyserial numpy scikit-learn joblib tensorflow pandas pyautogui
```

---

### 3. Collect Training Data

**Option A — Static Grid (Recommended for trackpad use)**

`rec_grid.py` guides you to place your hand at each of 225 grid points (15×15) and records averaged sensor readings per point.

```bash
python rec_grid.py
```

- A window will show a red dot at each target position.
- Move your hand to match the dot's position, then press **[Enter]** in the terminal.
- Output: `mpr121_static_grid_data01.csv`

**Option B — Dynamic Pattern Following**

`data_rec.py` animates a circle or line pattern, then records sensor data while you trace the same pattern with your finger.

```bash
python data_rec.py
```

- Choose pattern type (circle `1` or horizontal line `2`).
- Watch the preview, then follow the pattern with your hand.
- Output: `mpr121_user_pattern_data.csv`

**Option C — Raw Serial Logger**

`py_mat.py` is a minimal logger that records all 5-channel sensor data directly to CSV with timestamps.

```bash
python py_mat.py
```

- Output: `mpr121_error_data.csv`

---

### 4. Train the Model

Training scripts are located in the `Archive/` folder. The main pipeline (`train_trajectory_model.py`) trains the CNN-BiLSTM trajectory model:

```bash
cd Archive
python train_trajectory_model.py
```

This will:
1. Auto-detect CSV files in the data directory.
2. Extract and preprocess sequences.
3. Apply feature engineering and data augmentation.
4. Train the CNN-BiLSTM model with CPU optimization.
5. Save the trained model as `trajectory_model.h5`.

For the simpler Random Forest model, use the static grid data with scikit-learn's `RandomForestRegressor` targeting `Target_Grid_X` and `Target_Grid_Y` columns, then save with `joblib.dump`.

---

### 5. Run the Live Demo

**Recommended (Random Forest + EMA) — `aura_demo.py`**

```bash
python aura_demo.py
```

Before running, open `aura_demo.py` and set your baseline values:

```python
YOUR_BASELINES = np.array([-1.94, -1.36, -1.94, -1.16])  # replace with yours
```

Requires: `rf_model.pkl`, `scaler_X.pkl`

**Alternative (Kalman filter) — `inference.py`**

```bash
python inference.py
```

Requires: `cursor_model.pkl`

**Advanced (CNN-BiLSTM) — `emt_live.py`**

```bash
python emt_live.py
```

Requires: `trajectory_model.h5`

All demos open a **fullscreen black window** with a red dot that tracks your hand position. Press `Ctrl+C` in the terminal to stop.

---

## Pipeline Overview

```
MPR121 Sensor (5 ch)
        │
   Arduino Firmware
   (filtered + error values per channel)
        │  USB Serial @ 2 Mbps
        │
  ┌─────▼──────────────────────────────────┐
  │         Python Real-Time Loop          │
  │  1. Read serial  (~200 Hz)             │
  │  2. Median filter (window=3)           │
  │  3. Baseline subtraction               │
  │  4. Feature scaling (StandardScaler)   │
  │  5. ML Prediction → (x, y)            │
  │  6. EMA / Kalman smoothing             │
  │  7. Map to screen coordinates          │
  │  8. Update GUI dot (~30 FPS)           │
  └─────────────────────────────────────────┘
        │
  Tkinter fullscreen overlay
```

**Input features used** (4 of the 10 available values):
- `er0` — Channel 0 error
- `er1` — Channel 1 error
- `er3` — Channel 3 error
- `er4` — Channel 4 error

---

## Configuration

Key settings in each script:

| Parameter | Default | Description |
|---|---|---|
| `SERIAL_PORT` | `/dev/ttyACM0` | Serial port of the Arduino |
| `BAUD_RATE` | `2000000` | Serial baud rate |
| `NUM_CHANNELS` | `5` | Total MPR121 channels |
| `MEDIAN_FILTER_SIZE` | `3` | Window size for median filter |
| `EMA_ALPHA` | `0.25` | Smoothing factor (lower = smoother, more lag) |
| `GRID_X_MIN/MAX` | `1.0 / 13.0` | Grid coordinate range (X) |
| `GRID_Y_MIN/MAX` | `1.0 / 14.0` | Grid coordinate range (Y) |
| `UPDATE_LOOP_MS` | `33` | GUI update interval (~30 FPS) |
| `SERIAL_LOOP_MS` | `5` | Serial read interval (~200 Hz) |

> **Windows users**: Change `SERIAL_PORT` to `'COM3'` (or your actual port).

---

## Models

| File | Type | Used in | Description |
|---|---|---|---|
| `rf_model.pkl` | Random Forest | `aura_demo.py` | Predicts (x, y) from 4 scaled error features |
| `scaler_X.pkl` | StandardScaler | `aura_demo.py` | Feature normalizer for the RF model |
| `cursor_model.pkl` | Dict (RF + stats) | `inference.py` | Model bundled with normalization stats and grid dimensions |
| `trajectory_model.h5` | CNN-BiLSTM (TF/Keras) | `emt_live.py` | Sequence-to-trajectory deep learning model |

### CNN-BiLSTM Architecture (`emt_live.py`)

```
Input (500 timesteps × 48 features)
  → Multi-scale Conv1D (kernels: 3, 5, 7) + BatchNorm
  → Concatenate
  → BiLSTM (32 units) + Dropout
  → BiLSTM (16 units) + Dropout
  → SpatialAttention
  → TimeDistributed Dense (16, ReLU)
  → TimeDistributed Dense (2, tanh)   ← (x, y) per timestep
```

---

## File Reference

| File | Purpose |
|---|---|
| `Test_emt_copy/Test_emt_copy.ino` | Arduino firmware — reads MPR121, streams data over serial |
| `Test_emt_copy/Constants.h/.cpp` | Sensor configuration (baud rate, thresholds, filter settings) |
| `py_mat.py` | Minimal serial-to-CSV logger |
| `data_rec.py` | Dynamic pattern-following data recorder |
| `rec_grid.py` | Static 15×15 grid data recorder |
| `aura_demo.py` | Real-time demo: RF model + EMA smoothing |
| `inference.py` | Real-time demo: cursor_model + Kalman filter |
| `emt_live.py` | Real-time demo: CNN-BiLSTM model |
| `Archive/data_processing.py` | CSV → labeled sequences for model training |
| `Archive/feature_engineering.py` | Feature construction utilities |
| `Archive/augmentation.py` | Data augmentation for limited datasets |
| `Archive/train_trajectory_model.py` | Full CNN-BiLSTM training pipeline |

---

## License

See [LICENSE](LICENSE) for details.
