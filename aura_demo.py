import serial
import re
import numpy as np
import time
import tkinter as tk
from collections import deque
import joblib
import warnings

# --- Constants & Configuration (MODIFIED) ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 2000000
NUM_CHANNELS = 5 # Still 5 channels from serial, but we only use 4
MODEL_FILE = 'rf_model.pkl'       # <--- MODIFIED
SCALER_FILE = 'scaler_X.pkl'     # <--- NEW

# VVVVVV  ACTION REQUIRED VVVVVV
# Put your 4 baseline values from the .txt file here
# YOUR_BASELINES = np.array([-1.94, -1.36, -1.94, -1.16]) 
YOUR_BASELINES = np.array([-2, -2, -2, -2]) 


# --- Preprocessing & Smoothing (NEW) ---
MEDIAN_FILTER_SIZE = 3
EMA_ALPHA = 0.25 # Your requested value. See notes below.

# --- Grid & Screen Mapping (NEW) ---
# Based on your mapping formulas
GRID_X_MIN = 1.0
GRID_X_MAX = 13.0
GRID_Y_MIN = 1.0
GRID_Y_MAX = 14.0

PREVIEW_WIDTH = 0
PREVIEW_HEIGHT = 0
DOT_SIZE = 20
UPDATE_LOOP_MS = 33 # ~30 FPS for prediction loop
SERIAL_LOOP_MS = 5  # ~200 Hz for serial read loop

# --- Global Variables ---
ser = None
window = None
canvas = None
dot = None
model = None    # <--- MODIFIED
scaler = None   # <--- MODIFIED
baselines = None # <--- NEW

# Use a deque for the median filter
data_buffer = deque(maxlen=MEDIAN_FILTER_SIZE) # <--- MODIFIED

# EMA state variables
smooth_x = (GRID_X_MIN + GRID_X_MAX) / 2.0 # <--- NEW
smooth_y = (GRID_Y_MIN + GRID_Y_MAX) / 2.0 # <--- NEW

# --- KalmanFilter class (REMOVED) ---
# This is no longer needed and has been replaced by EMA.

def read_serial_data():
    """
    Reads data from serial, parses the 4 required channels,
    and appends them to the data_buffer.
    """
    global data_buffer, ser

    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            if not line:
                continue
            
            # Find all "Channel X: (val), error :(err)" matches
            matches = re.findall(r"Channel \d+: (\d+), error :(-?\d+)", line)
            
            if len(matches) == NUM_CHANNELS:
                try:
                    # We get 5 channels, 2 values each (filtered, error)
                    # This creates a flat list of 10 values
                    raw_features_10 = []
                    for match in matches:
                        raw_features_10.append(float(match[0]))  # Filtered
                        raw_features_10.append(float(match[1]))  # Error
                    
                    # --- Input Definition (MODIFIED) ---
                    # Your model requires [er0, er1, er3, er4]
                    # Index 1 = ch0 error
                    # Index 3 = ch1 error
                    # Index 7 = ch3 error
                    # Index 9 = ch4 error
                    frame = np.array([
                        raw_features_10[1], # er0
                        raw_features_10[3], # er1
                        raw_features_10[7], # er3
                        raw_features_10[9]  # er4
                    ])
                    # --- End Input Definition ---

                    data_buffer.append(frame)
                    
                except (ValueError, IndexError) as e:
                    print(f"Data parsing error: {e}, Line: {line}")
                    pass

    except Exception as e:
        print(f"Serial read error: {e}")

    if window:
        window.after(SERIAL_LOOP_MS, read_serial_data)

def update_gui_prediction():
    """
    Runs the full preprocessing pipeline on the latest data,
    predicts, smooths, and updates the GUI dot.
    """
    # <--- THIS ENTIRE FUNCTION IS REWRITTEN ---
    global data_buffer, canvas, dot, model, scaler, baselines
    global smooth_x, smooth_y, PREVIEW_WIDTH, PREVIEW_HEIGHT

    # Wait until the median filter buffer is full
    if len(data_buffer) < MEDIAN_FILTER_SIZE:
        # print(f"Filling buffer... {len(data_buffer)}/{MEDIAN_FILTER_SIZE}", end='\r')
        if window:
            window.after(UPDATE_LOOP_MS, update_gui_prediction)
        return
    
    try:
        # --- 1. Median Filtering ---
        samples = np.array(data_buffer)
        median_filtered_sample = np.median(samples, axis=0)
        
        # --- 2. Baseline Subtraction ---
        subtracted_sample = median_filtered_sample - baselines

        # --- 3. Feature Scaling ---
        # Reshape to (1, 4) for the scaler
        scaled_sample = scaler.transform(subtracted_sample.reshape(1, -1))

        # --- 4. Model Prediction ---
        # Model should output (x, y)
        # We expect shape (1, 2) from predict
        pred_xy = model.predict(scaled_sample)
        pred_x = pred_xy[0, 0]
        pred_y = pred_xy[0, 1]
        
        # --- 5. Exponential Moving Average (EMA) Smoothing ---
        smooth_x = (EMA_ALPHA * pred_x) + (1 - EMA_ALPHA) * smooth_x
        smooth_y = (EMA_ALPHA * pred_y) + (1 - EMA_ALPHA) * smooth_y

        # --- 6. Cursor Mapping ---
        # Map from grid coordinates (1-13, 1-14) to screen pixels
        screen_x = (smooth_x - GRID_X_MIN) / (GRID_X_MAX - GRID_X_MIN) * PREVIEW_WIDTH
        
        # Invert Y-axis as per your formula: (1.0 - ...)
        screen_y = (1.0 - (smooth_y - GRID_Y_MIN) / (GRID_Y_MAX - GRID_Y_MIN)) * PREVIEW_HEIGHT

        # Clip to ensure dot stays on screen
        x_pos = np.clip(screen_x, 0, PREVIEW_WIDTH)
        y_pos = np.clip(screen_y, 0, PREVIEW_HEIGHT)

        if canvas:
            canvas.coords(
                dot,
                x_pos - DOT_SIZE / 2, y_pos - DOT_SIZE / 2,
                x_pos + DOT_SIZE / 2, y_pos + DOT_SIZE / 2
            )
            
    except Exception as e:
        print(f"\nPrediction loop error: {e}")
        # Keep trying
    
    if window:
        window.after(UPDATE_LOOP_MS, update_gui_prediction)


def load_dependencies():
    """
    Loads all required files (model, scaler, baselines)
    into global variables.
    """
    # <--- THIS IS A NEW HELPER FUNCTION ---
    global model, scaler, baselines
    
    try:
        # --- 1. Load Baselines ---
        print("Loading baselines...")
        baselines = YOUR_BASELINES
        if baselines.shape != (4,) or np.all(baselines == 0):
            print(f"--- WARNING: Baselines are {baselines} ---")
            print(">>> PLEASE UPDATE the 'YOUR_BASELINES' variable! <<<")
            time.sleep(2)
        print(f"Baselines loaded: {baselines}")

        # --- 2. Load Scaler ---
        print(f"Loading feature scaler from '{SCALER_FILE}'....")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = joblib.load(SCALER_FILE)
        print("Scaler loaded successfully.")
        
        # --- 3. Load Model ---
        print(f"Loading ML model from '{MODEL_FILE}'....")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(MODEL_FILE)
        print("Model loaded successfully.")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n--- FATAL ERROR: File not found ---")
        print(f"Could not find: {e.filename}")
        print("Please make sure rf_model.pkl and scaler_X.pkl are in the same folder.")
        return False
    except Exception as e:
        print(f"\n--- FATAL ERROR: Could not load dependencies ---")
        print(f"Error: {e}")
        return False

def main():
    """
    Main execution function.
    """
    # <--- MODIFIED to use new load_dependencies function ---
    global ser, window, canvas, dot, PREVIEW_WIDTH, PREVIEW_HEIGHT

    print("--- Starting Aura Real-Time Pipeline ---")
    
    # --- 1. Load Model, Scaler, Baselines ---
    if not load_dependencies():
        exit()

    # --- 2. Connect Serial ---
    try:
        print(f"Connecting to port '{SERIAL_PORT}'...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer() # Clear any old, junk data
        time.sleep(0.1)
        print("Connection successful!")
    except serial.SerialException as e:
        print(f"\n--- FATAL ERROR: Could not open {SERIAL_PORT} ---")
        print("Is the device plugged in? Is the port correct?")
        exit()

    # --- 3. Setup GUI ---
    try:
        print("Opening preview window...")
        window = tk.Tk()
        window.title("Aura Live Demo")

        PREVIEW_WIDTH = window.winfo_screenwidth()
        PREVIEW_HEIGHT = window.winfo_screenheight()
    
        window.attributes('-fullscreen', True)
        canvas = tk.Canvas(window, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg='black')
        canvas.pack()

        start_x = PREVIEW_WIDTH / 2
        start_y = PREVIEW_HEIGHT / 2
        dot = canvas.create_oval(
            start_x - DOT_SIZE / 2, start_y - DOT_SIZE / 2,
            start_x + DOT_SIZE / 2, start_y + DOT_SIZE / 2,
            fill='red', outline='red'
        )
        print("GUI ready. Move your hand over the sensor!")
        
        # --- 4. Start the loops ---
        read_serial_data()
        update_gui_prediction()
        
        window.mainloop()
        
    except Exception as e:
        print(f"\nA fatal GUI error occurred: {e}")

    finally:
        if window:
            try: window.destroy()
            except: pass
        if ser and ser.is_open:
            ser.close()
            print("Serial port closed.")
        print("\nScript terminated.")


if __name__ == "__main__":
    main()