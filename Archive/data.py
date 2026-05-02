import serial
import csv
import re
import numpy as np
import tensorflow as tf
import time
import math
import datetime
import tkinter as tk
from collections import deque
from feature_engineering import RobustFeatureEngineer
from models.trajectory_only_model import SpatialAttentionLayer
import joblib
from pathlib import Path
import re

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 2000000
# --- Corrected paths to look inside the sub-directory ---
MODEL_FILE = 'trajectory_model.h5'
SCALER_FILE = 'trajectory_model_complete.pkl'
NUM_CHANNELS = 5

# --- Model & GUI Settings ---
SEQ_LENGTH = 500
NUM_FEATURES = 46 # 10 sensor + 2 mouse + 36 derived = 48

NUM_RAW_FEATURES = NUM_CHANNELS * 2
PREVIEW_WIDTH = 600
PREVIEW_HEIGHT = 400
DOT_SIZE = 10

# --- Global Variables ---
ser = None
window = None
canvas = None
dot = None
raw_data_buffer = deque(maxlen=SEQ_LENGTH)
feature_engineer = None

# --- NEW Logging Globals ---
LOG_FILE_PREFIX = 'model_input_log' # <-- NEW: Filename prefix for logs
log_counter = 0                     # <-- NEW: Counter for log filenames
save_log_flag = False               # <-- NEW: Flag to signal a log save


def preprocess_data(raw_sequence_list):
    """
    Uses the loaded RobustFeatureEngineer to convert the (500, 10) raw data
    into (500, 48) features for the model.
    """
    global feature_engineer 

    # 1. Pad 10 features to 12 (for the engineer)
    raw_sequence_padded = [row + [0.0, 0.0] for row in raw_sequence_list]

    # 2. Convert to a (1, 500, 12) numpy array
    raw_sequence_np = np.array(raw_sequence_padded).reshape(1, SEQ_LENGTH, 12)

    # 3. Call the fitted processor's 'process_features' method.
    processed_sequence, _ = feature_engineer.process_features(raw_sequence_np, fit=False, verbose=False)

    # 4. Return the (500, 48) feature array
    return processed_sequence[0]

def read_serial_data():
    """
    Non-blocking function to read from serial and add to the buffer.
    """
    global raw_data_buffer, ser
    
    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue

            matches = re.findall(r"Channel \d+: (\d+), error :(-?\d+)", line)

            if len(matches) == 5:
                row_data = []
                try:
                    for match in matches:
                        row_data.append(float(match[0]))
                        row_data.append(float(match[1])) 

                    raw_data_buffer.append(row_data)
                    # --- Optional Logging ---
                    if len(raw_data_buffer) % 50 == 0 and len(raw_data_buffer) < SEQ_LENGTH:
                        print(f"Buffer filling: {len(raw_data_buffer)} / {SEQ_LENGTH}")
                    # ---------------------------
                except ValueError: 
                    print(f"Serial parse error: Could not convert values in line: {line}")
    except Exception as e:
        print(f"Serial read error: {e}")

    # Add check to prevent error if window is closed
    if window:
        window.after(5, read_serial_data)


def update_prediction(model):
    """
    Runs periodically to take the data buffer, preprocess it, predict, and move the dot.
    """
    global raw_data_buffer, canvas, dot
    global save_log_flag, log_counter # <-- NEW: Access logging globals
    
    if len(raw_data_buffer) < SEQ_LENGTH:
        print(f"Waiting for buffer... {len(raw_data_buffer)} / {SEQ_LENGTH}", end='\r')
        if window:
            window.after(33, update_prediction, model) # ~30 FPS
        return

    # 1. Get the raw data sequence
    raw_sequence = list(raw_data_buffer)
    
    # 2. Preprocess the data
    processed_sequence = preprocess_data(raw_sequence)
    
    # 3. Reshape for the model
    model_input = processed_sequence.reshape(1, SEQ_LENGTH, NUM_FEATURES)
    
    # 4. Predict
    prediction_sequence = model.predict(model_input, verbose=0)
    
    # --- NEW LOGGING BLOCK ---
    if save_log_flag:
        try:
            filename = f"{LOG_FILE_PREFIX}_{log_counter:04d}.npz"
            # Save the input AND the full predicted trajectory
            np.savez_compressed(
                filename,
                model_input=model_input[0], # Saves the (500, 46) input
                model_output_sequence=prediction_sequence[0] # Saves the (500, 2) output
            )
            print(f"\n--- SUCCESS: Saved input/output snapshot to {filename} ---")
            log_counter += 1
        except Exception as e:
            print(f"\n--- ERROR: Could not save log file: {e} ---")
        finally:
            save_log_flag = False # Reset the flag
    # --- END NEW LOGGING BLOCK ---

    # 5. Get the last (most recent) (x, y) coordinate
    predicted_xy = prediction_sequence[0][-1]
    
    # 6. Scale the output
    x_pos = (predicted_xy[0] + 1) / 2 * PREVIEW_WIDTH
    y_pos = (predicted_xy[1] + 1) / 2 * PREVIEW_HEIGHT
    
    # Print the coordinates to the console
    print(f"Model Output (raw): [{predicted_xy[0]:.3f}, {predicted_xy[1]:.3f}]  |  Screen Coords (X, Y): [{x_pos:.1f}, {y_pos:.1f}]")
    
    # 7. Move the dot
    if canvas:
        canvas.coords(
            dot,
            x_pos - DOT_SIZE / 2,
            y_pos - DOT_SIZE / 2,
            x_pos + DOT_SIZE / 2,
            y_pos + DOT_SIZE / 2
        )
    
    if window:
        window.after(33, update_prediction, model) # ~30 FPS


# --- Main Script ---
try:
    # 1. Load the Keras Model
    print(f"Loading model '{MODEL_FILE}'...")
    model = tf.keras.models.load_model(
        MODEL_FILE,
        custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer},
        compile=False
    )
    model.summary()
    print("Model loaded successfully.")

    # 2. Load the FITTED Feature Engineer
    print(f"Loading fitted feature engineer from '{SCALER_FILE}'...")
    loaded_data = joblib.load(SCALER_FILE)
    print("File loaded. Accessing nested feature engineer object...")
    try:
        scalers_dict = loaded_data['scalers']
        try:
            feature_engineer = scalers_dict['feature_engineer']
            if not hasattr(feature_engineer, 'process_features'):
                print(f"\n--- FATAL ERROR ---")
                print(f"Found key 'scalers' -> 'feature_engineer', but it's not the right object.")
                print(f"Object type found: {type(feature_engineer)}")
                print("Expected an object with a 'process_features' method.")
                raise TypeError("Loaded object is not a RobustFeatureEngineer.")
        except KeyError:
            print(f"\n--- FATAL ERROR ---")
            print(f"Found 'scalers' dictionary, but could not find 'feature_engineer' inside it.")
            print(f"Keys inside 'scalers' are: {scalers_dict.keys()}")
            raise
    except KeyError:
        print(f"\n--- FATAL ERROR ---")
        print(f"Could not find the top-level key 'scalers' in the .pkl file.")
        print(f"Available top-level keys are: {loaded_data.keys()}")
        raise
    print("Feature engineer object loaded successfully.")

    # 3. Connect to Serial Port
    print(f"Attempting to connect to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    print("Connection successful!")
    
    # 4. Set up the GUI
    print("Opening preview window...")
    window = tk.Tk()
    window.title("Live Trajectory Demo")
    window.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
    
    canvas = tk.Canvas(window, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg='black')
    canvas.pack()
    
    dot = canvas.create_oval(
        (PREVIEW_WIDTH/2) - DOT_SIZE / 2,
        (PREVIEW_HEIGHT/2) - DOT_SIZE / 2,
        (PREVIEW_WIDTH/2) + DOT_SIZE / 2,
        (PREVIEW_HEIGHT/2) + DOT_SIZE / 2,
        fill='red', 
        outline='red'
    )
    
    print("GUI ready. Waiting for data buffer to fill...")
    
    # --- NEW: Key binding for logging ---
    def set_log_flag(event):
        global save_log_flag
        # Check for the 'l' key (lowercase)
        if event.keysym == 'l':
            print("\n--- LOG REQUESTED (Press 'L'): Will save on next frame ---")
            save_log_flag = True
    
    window.bind("<KeyPress-l>", set_log_flag) # Bind the 'L' key
    print("\n*******************************************************")
    print(">>> Press the 'L' key to save an input/output snapshot <<<")
    print("*******************************************************\n")
    # --- END NEW ---
    
    # 5. Start the loops
    read_serial_data()
    update_prediction(model)
    window.mainloop()
    

except FileNotFoundError as e:
    print(f"\n--- FATAL ERROR: FILE NOT FOUND ---")
    print(f"Could not find a required file.")
    print(f"Details: {e}")
    print(f"Please make sure '{MODEL_FILE}' and '{SCALER_FILE}' are in the 'trajectory_model/' sub-directory.")
except serial.serialutil.SerialException as e:
    print(f"\n--- FATAL ERROR ---")
    print(f"Could not open serial port '{SERIAL_PORT}'.")
    print("Please check the port and ensure you have permissions.")
    print(f"Original error: {e}")
except KeyboardInterrupt:
    print("\nStopping demo.")
except Exception as e:
    print(f"A fatal error occurred: {e}")
finally:
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
    if window:
        try:
            window.destroy()
            print("Window closed.")
        except tk.TclError:
            pass # Window already closed