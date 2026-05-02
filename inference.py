import serial
import re
import numpy as np
import time
import tkinter as tk
from collections import deque
import joblib
import warnings

SERIAL_PORT = '/dev/ttyACM1'
BAUD_RATE = 2000000
NUM_CHANNELS = 5
MODEL_FILE = 'cursor_model.pkl'
PREVIEW_WIDTH = 0
PREVIEW_HEIGHT = 0
DOT_SIZE = 20
UPDATE_LOOP_MS = 33
SERIAL_LOOP_MS = 5

ser = None
window = None
canvas = None
dot = None
model_dict = None
kalman_filter = None

data_buffer = deque(maxlen=4)
norm_stats_dict = {}
feature_cols_16 = []
base_feature_names_4 = []
BOX_WIDTH_CM = 14.0
BOX_HEIGHT_CM = 15.0

class KalmanFilter:
    def __init__(self, dt=0.02, process_var=0.5, meas_var=4.0):
        self.dt = dt
        self.F = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1,0],
                           [0,0,0,1]])
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]])
        self.Q = np.eye(4) * process_var
        self.R = np.eye(2) * meas_var
        self.x = np.zeros((4,1))
        self.P = np.eye(4) * 1000.0

    def reset(self, x0, y0):
        self.x = np.array([[x0],[y0],[0.0],[0.0]])
        self.P = np.eye(4) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z).reshape(2,1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x[0,0], self.x[1,0]
def read_serial_data():
    global data_buffer, ser

    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            if not line:
                continue
            matches = re.findall(r"Channel \d+: (\d+), error :(-?\d+)", line)
            if len(matches) == NUM_CHANNELS:
                try:
                    raw_features_10 = []
                    for match in matches:
                        raw_features_10.append(float(match[0]))  # Filtered
                        raw_features_10.append(float(match[1]))  # Error

                        frame = np.array([
                            raw_features_10[0], #CH0 filtered
                            raw_features_10[3], # CH1 Error
                            raw_features_10[7], # Ch3 Error
                            raw_features_10[8] # Ch4 filtered
                        ])

                        data_buffer.append(frame)
                except(ValueError, IndexError):
                    pass

    except Exception as e:
        print(f"Serial read error: {e}")

    if window:
        window.after(SERIAL_LOOP_MS, read_serial_data)

def update_gui_prediction():
    global data_buffer, canvas, dot, model_dict, kalman_filter
    global norm_stats_dict, feature_cols_16, base_feature_names_4

    if len(data_buffer) <= 4:
        print(f"Filling buffer... {len(data_buffer)}/4", end='\r')
        if window:
            window.after(UPDATE_LOOP_MS, update_gui_prediction)
        return
    
    try:
        unscaled_feature_vector_16 = np.concatenate([
            data_buffer[3],
            data_buffer[2],
            data_buffer[1],
            data_buffer[0]
        ])
    except IndexError:
        if window:
            window.after(UPDATE_LOOP_MS, update_gui_prediction)
            return
        
    print(f"Raw Input [t]: {unscaled_feature_vector_16[0]:.1f}, {unscaled_feature_vector_16[1]:.1f}, {unscaled_feature_vector_16[2]:.1f}, {unscaled_feature_vector_16[3]:.1f}", end='\r')

    scaled_feature_vector_16 = np.zeros(16)

    try:
        for i in range(16):
            full_feature_name = feature_cols_16[i]
            base_name_found = ""
            for base_name in base_feature_names_4:
                if full_feature_name.startswith(base_name):
                    base_name_found = base_name
                    break
            stats = norm_stats_dict[base_name_found]
            median = stats['med']
            mad = stats['mad']

            raw_val = unscaled_feature_vector_16[i]
            scaled_feature_vector_16[i] = (raw_val - median) / (mad + 1e-6)
    except Exception as e:
        print(f"\nCRITICAL ERROR during normalization: {e}")
        if window: window.destroy()
        return
    scaled_features_ready = scaled_feature_vector_16.reshape(1, -1)

    try:
        x_cm_pred = model_dict['reg_x'].predict(scaled_features_ready)[0]
        y_cm_pred = model_dict['reg_y'].predict(scaled_features_ready)[0]
    except Exception as e:
        print(f"\nPrediction Error: {e}")
        if window: window.after(UPDATE_LOOP_MS, update_gui_prediction)
        return
    
    kalman_filter.predict()
    filtered_xy = kalman_filter.update(np.array([x_cm_pred, y_cm_pred]))

    x_cm_filtered = filtered_xy[0]
    y_cm_filtered = filtered_xy[1]

    x_pos = (x_cm_filtered/ BOX_WIDTH_CM) * PREVIEW_WIDTH
    y_pos = (y_cm_filtered/ BOX_HEIGHT_CM) * PREVIEW_HEIGHT

    x_pos =np.clip(x_pos, 0, PREVIEW_WIDTH)
    y_pos =np.clip(y_pos, 0, PREVIEW_HEIGHT)

    if canvas:
        canvas.coords(
            dot,
            x_pos - DOT_SIZE / 2, y_pos - DOT_SIZE / 2,
            x_pos + DOT_SIZE / 2, y_pos + DOT_SIZE / 2
        )
    if window:
        window.after(UPDATE_LOOP_MS, update_gui_prediction)


    try:
        print(f"Loading ML model from '{MODEL_FILE}'....")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_dict = joblib.load(MODEL_FILE)
        print("Model loaded successfully.")

        if 'norm_stats' in model_dict and \
            'feature_cols' in model_dict and \
            'base_features' in model_dict:

            print(" -> Found 'norm_stats', 'feature_cols', and 'base_features' in the model.")

            norm_stats_dict = model_dict['norm_stats']
            feature_cols_16 = model_dict['feature_cols']
            base_feature_names_4 = model_dict['base_features']
            BOX_WIDTH_CM = model_dict.get('grid_w', 14.0)
            BOX_HEIGHT_CM = model_dict.get('grid_h', 15.0)

            # Sanity checks
            if len(feature_cols_16) != 16: ...
            if len(base_feature_names_4) != 4: ...
            print("  -> Model pipeline components loaded successfully.")

        else:
            # Fails if the .pkl file is missing keys
            print(f"--- FATAL ERROR: Model .pkl is missing critical keys! ---")
            exit()

        # --- 3. Initialize Kalman Filter ---
        kalman_filter = KalmanFilter(
            dt = UPDATE_LOOP_MS / 1000.0, # e.g., 0.033 seconds
            process_noise = 1,            # Your aggressive tuning value
            measurement_noise = 0.005     # Your aggressive tuning value
        )
        print("  -> Kalman filter initialized.")

        # --- 4. Connect Serial ---
        print(f"Connecting to port '{SERIAL_PORT}'...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.reset_input_buffer() # Clear any old, junk data
        time.sleep(0.1)
        print("Connection successful!")

        # --- 5. Setup GUI ---
        print("Opening preview window...")
        window = tk.Tk()
        window.title("Aura Live Demo")

        # Get your actual screen size
        PREVIEW_WIDTH = window.winfo_screenwidth()
        PREVIEW_HEIGHT = window.winfo_screenheight()
    
        window.attributes('-fullscreen', True) # Make it fullscreen

        canvas = tk.Canvas(window, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg='black')
        canvas.pack()

        # Create the red dot in the center
        start_x = PREVIEW_WIDTH / 2
        start_y = PREVIEW_HEIGHT / 2
        dot = canvas.create_oval(
            start_x - DOT_SIZE / 2, start_y - DOT_SIZE / 2,
            start_x + DOT_SIZE / 2, start_y + DOT_SIZE / 2,
            fill='red', outline='red'
        )

        print("GUI ready. Move your hand over the sensor!")
        
        # --- 6. Start the two loops ---
        read_serial_data()      # Kicks off the first serial read
        update_gui_prediction() # Kicks off the first GUI update
        
        # This blocks the script and keeps the GUI window open
        window.mainloop()
    # --- Error Handling & Cleanup ---
    except serial.SerialException as e:
        print(f"\n--- FATAL ERROR: Could not open {SERIAL_PORT} ---")
    except FileNotFoundError as e:
        print(f"\n--- FATAL ERROR: Could not find '{MODEL_FILE}' ---")
    except Exception as e:
        print(f"\nA fatal error occurred: {e}")

    finally:
    # This code runs NO MATTER WHAT (even if it crashes)
    # to ensure the program exits cleanly.
        if 'window' in locals() and window:
            try: window.destroy()
            except: pass
        if 'ser' in locals() and ser and ser.is_open:
            ser.close()
        print("\nScript terminated.")