import serial
import csv
import re
import numpy as np
import time
import math
import datetime
import tkinter as tk
from collections import deque
import re
import joblib  # For loading the model

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 2000000
NUM_CHANNELS = 5
MODEL_FILE = 'sensor_model_pipeline.pkl'  # Load our trained model

# --- GUI Settings ---
SEQ_LENGTH = 1  # We only care about the most recent value
PREVIEW_WIDTH = 600
PREVIEW_HEIGHT = 400
DOT_SIZE = 10

# --- Global Variables ---
ser = None
window = None
canvas = None
dot = None
raw_data_buffer = deque(maxlen=SEQ_LENGTH)  # We only need the last item
model = None  # This will hold our loaded model

# --- Min/Max values from your training data (X: 0-40, Y: 0-20) ---
# We use these to scale the model's output to the window size
DATA_X_MIN = 0
DATA_X_MAX = 40
DATA_Y_MIN = 0
DATA_Y_MAX = 20
# -----------------------------------------------------------------


def get_model_prediction(raw_data):
    """
    Uses the trained ML model to predict (X, Y) from the 4 sensor values.
    This version has the CORRECTED Y-AXIS MAPPING.
    """
    global model

    # raw_data is a list of 10 values:
    # [ch0_f, ch0_e, ch1_f, ch1_e, ch2_f, ch2_e, ch3_f, ch3_e, ch4_f, ch4_e]

    try:
        # --- 1. Get Sensor FILTERED values ---
        # We must match the *exact* order from the training script:
        # X_cols = ['Front(2)', 'Right(4)', 'Down(5)', 'Left(1)']
        
        # We map these names to your live channels (0-4)
        # using your provided image (image_6262b7.png) and CSV data type.
        
        # 'Front(2)' -> "Front" -> Channel 4 -> Filtered Value
        val_front = raw_data[8]
        
        # 'Right(4)' -> "Right" -> Channel 1 -> Filtered Value
        val_right = raw_data[2]
        
        # 'Down(5)' -> "Down" -> Channel 2 -> Filtered Value
        val_down = raw_data[4]
        
        # 'Left(1)' -> "Left" -> Channel 3 -> Filtered Value
        val_left = raw_data[6]


        # --- 2. Create the input array for the model ---
        # The order MUST be [Front, Right, Down, Left]
        features = np.array([[
            val_front,
            val_right,
            val_down,
            val_left
        ]])

        # --- 3. Predict ---
        # The pipeline handles scaling and prediction
        predicted_xy = model.predict(features)[0]  # Get the [X, Y] array
        raw_x = predicted_xy[0]
        raw_y = predicted_xy[1]

        # --- 4. Re-scale the model's output to window coordinates ---
        
        # X-Axis (Model X: 0-40 -> Window X: 0-600)
        x_pos = (raw_x - DATA_X_MIN) / (DATA_X_MAX - DATA_X_MIN) * PREVIEW_WIDTH
        
        # Y-Axis (INVERTED)
        # Model Y: 0 (Down) -> Window Y: 400 (Bottom)
        # Model Y: 20 (Up)  -> Window Y: 0 (Top)
        y_pos = PREVIEW_HEIGHT - ( (raw_y - DATA_Y_MIN) / (DATA_Y_MAX - DATA_Y_MIN) * PREVIEW_HEIGHT )

        # Clip values to ensure they stay inside the window
        x_pos = np.clip(x_pos, 0, PREVIEW_WIDTH)
        y_pos = np.clip(y_pos, 0, PREVIEW_HEIGHT)

        return [x_pos, y_pos], [raw_x, raw_y]

    except IndexError:
        return [PREVIEW_WIDTH / 2, PREVIEW_HEIGHT / 2], [0, 0]  # Default to center
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return [PREVIEW_WIDTH / 2, PREVIEW_HEIGHT / 2], [0, 0]


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

                    raw_data_buffer.append(row_data)  # Add the [10 values] list

                except ValueError:
                    print(f"Serial parse error: Could not convert values in line: {line}")
    except Exception as e:
        print(f"Serial read error: {e}")

    # Add check to prevent error if window is closed
    if window:
        window.after(5, read_serial_data)


def update_prediction():
    """
    Runs periodically to take the buffer, calculate position, and move the dot.
    """
    global raw_data_buffer, canvas, dot

    if not raw_data_buffer:
        # Waiting for the first-ever reading
        print(f"Waiting for first serial reading...", end='\r')
        if window:
            window.after(33, update_prediction)  # ~30 FPS
        return

    # 1. Get the single most recent reading
    raw_sequence = list(raw_data_buffer)
    raw_data = raw_sequence[0]  # This is our list of 10 features

    # 2. Calculate position using our new function
    (x_pos, y_pos), (raw_x, raw_y) = get_model_prediction(raw_data)

    # Print the coordinates to the console
    print(f"Model Raw (X,Y): [{raw_x:6.1f}, {raw_y:6.1f}] | Screen (X,Y): [{x_pos:6.1f}, {y_pos:6.1f}]", end='\r')

    # 4. Move the dot
    if canvas:
        canvas.coords(
            dot,
            x_pos - DOT_SIZE / 2,
            y_pos - DOT_SIZE / 2,
            x_pos + DOT_SIZE / 2,
            y_pos + DOT_SIZE / 2
        )

    if window:
        window.after(33, update_prediction)  # ~30 FPS


# --- Main Script ---
try:
    # --- 1. Load the Trained Model ---
    print(f"Loading trained model from '{MODEL_FILE}'...")
    try:
        model = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"--- FATAL ERROR: FILE NOT FOUND ---")
        print(f"Could not find '{MODEL_FILE}'.")
        print("Please run the 'train_sensor_model.py' script first,")
        print("and copy the '.pkl' file to this folder.")
        exit()
    print("Model loaded successfully.")

    # 2. Connect to Serial Port
    print(f"Attempting to connect to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    print("Connection successful!")

    # 3. Set up the GUI
    print("Opening preview window...")
    window = tk.Tk()
    window.title("Live Trajectory Demo (ML Model)")
    window.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")

    canvas = tk.Canvas(window, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg='black')
    canvas.pack()

    dot = canvas.create_oval(
        (PREVIEW_WIDTH / 2) - DOT_SIZE / 2,
        (PREVIEW_HEIGHT / 2) - DOT_SIZE / 2,
        (PREVIEW_WIDTH / 2) + DOT_SIZE / 2,
        (PREVIEW_HEIGHT / 2) + DOT_SIZE / 2,
        fill='red',
        outline='red'
    )

    print("GUI ready. Waiting for data...")

    # 4. Start the loops
    read_serial_data()
    update_prediction()  # No 'model' argument needed
    window.mainloop()


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
            pass  # Window already closed