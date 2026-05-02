import serial
import csv
import re
import numpy as np  # We still use numpy for 'clip'
import time
import math
import datetime
import tkinter as tk
from collections import deque
import re

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 2000000
NUM_CHANNELS = 5
kp_x=25
kp_y=25


# --- !!! TUNE THIS VALUE !!! ---
# This is the "sensitivity" knob.
# Start with 1000.
# - If the dot moves *too fast* or is always "stuck" at the edges,
#   make this number *LARGER* (e.g., 1500, 2000).
# - If the dot moves *too slow* or never reaches the edges,
#   make this number *SMALLER* (e.g., 800, 500).
MAX_SENSOR_DIFFERENCE = 1000.0

# --- GUI Settings ---
SEQ_LENGTH = 1 # We only care about the most recent value
# These will be updated to your actual screen size
PREVIEW_WIDTH = 600
PREVIEW_HEIGHT = 400
DOT_SIZE = 10

# --- Global Variables ---
ser = None
window = None
canvas = None
dot = None
raw_data_buffer = deque(maxlen=SEQ_LENGTH) # We only need the last item

# --- NO MODELS OR SCALERS NEEDED ---


def calculate_xy_position(raw_data):
    """
    Directly maps 10 sensor inputs to a [-1, 1] (x, y) coordinate.
    This version is CORRECTED to use the channel map from your image
    and proper normalization.
    """
    
    # raw_data is a list of 10 values:
    # [ch0_f, ch0_e, ch1_f, ch1_e, ch2_f, ch2_e, ch3_f, ch3_e, ch4_f, ch4_e]
    
    # --- CORRECT Channel Mapping (based on your image image_6262b7.png) ---
    # We use the ERROR values (at odd indices)
    
    # left_val  (Image says Left(3)) -> Channel 3 Error
    left_val = raw_data[1]   
    
    # right_val (Image says Right(1)) -> Channel 1 Error
    right_val = raw_data[7]  
    
    # upper_val (Image says "front" = Front(4)) -> Channel 4 Error
    upper_val = raw_data[3]  
    
    # lower_val (Image says "down" = Down(2)) -> Channel 2 Error
    lower_val = raw_data[9]  

    try:
        # --- X-Axis Logic ---
        # (Right - Left)
        x_raw = right_val - left_val
        
        # --- Y-Axis Logic ---
        # (Down - Up/Front)
        y_raw = lower_val - upper_val
        
        # --- Normalization ---
        # Divide by our "max" value to get a number between -1.0 and 1.0
        x_norm = x_raw / MAX_SENSOR_DIFFERENCE
        y_norm = y_raw / MAX_SENSOR_DIFFERENCE
        
        # --- Clip ---
        # Use np.clip to make sure the value *never* goes
        # outside the -1.0 to 1.0 range.
        x_norm_clipped = np.clip(x_norm, -1.0, 1.0)*kp_x
        y_norm_clipped = np.clip(y_norm, -1.0, 1.0)*kp_y
        if(x_norm_clipped<=0.075 and x_norm_clipped>=-0.075):x_norm_clipped=0
        if(y_norm_clipped<=-0.10 and y_norm_clipped>=0.10):y_norm_clipped=0
        
        # This function should ONLY return the [-1.0, 1.0] value.
        # The scaling to pixels happens in update_prediction()
        return [x_norm_clipped, y_norm_clipped]

    except IndexError:
        # This might happen if the buffer is empty on the first frame
        return [0.0, 0.0] # Default to center
    except Exception as e:
        print(f"Error in calculate_xy: {e}")
        return [0.0, 0.0]


def read_serial_data():
    """
    Non-blocking function to read from serial and add to the buffer.
    (This function is unchanged)
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

                    raw_data_buffer.append(row_data) # Add the [10 values] list
                    
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
    REMOVED 'model' argument.
    """
    global raw_data_buffer, canvas, dot
    
    if not raw_data_buffer:
        # Waiting for the first-ever reading
        print(f"Waiting for first serial reading...", end='\r')
        if window:
            window.after(33, update_prediction) # ~30 FPS
        return

    # 1. Get the single most recent reading
    raw_sequence = list(raw_data_buffer) 
    raw_data = raw_sequence[0] # This is our list of 10 features
    
    # 2. Calculate position using our new function
    predicted_xy = calculate_xy_position(raw_data)
    
    # 3. Scale the [-1, 1] output to screen coordinates
    #    This uses the global PREVIEW_WIDTH/HEIGHT, which are now
    #    set to your full screen size.
    x_pos = (predicted_xy[0] + 1) / 2 * PREVIEW_WIDTH
    y_pos = (predicted_xy[1] + 1) / 2 * PREVIEW_HEIGHT
    
    # Print the coordinates to the console
    print(f"Raw Diff (X, Y): [{predicted_xy[0]:.3f}, {predicted_xy[1]:.3f}]  |  Screen Coords (X, Y): [{x_pos:.1f}, {y_pos:.1f}]", end='\r')
    
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
        window.after(33, update_prediction) # ~30 FPS


# --- Main Script ---
try:
    # --- NO MODEL LOADING NEEDED ---
    print("Starting direct-map.")

    # 3. Connect to Serial Port
    print(f"Attempting to connect to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    print("Connection successful!")
    
    # 4. Set up the GUI
    print("Opening preview window...")
    window = tk.Tk()
    window.title("Live Trajectory Demo (Direct Map)")

    # --- FULLSCREEN MODIFICATION ---
    # We must modify the globals so the scaling in update_prediction() works
    # global PREVIEW_WIDTH, PREVIEW_HEIGHT
    PREVIEW_WIDTH = window.winfo_screenwidth()
    PREVIEW_HEIGHT = window.winfo_screenheight()
    
    print(f"Detected screen size: {PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")

    # Set to fullscreen
    window.attributes('-fullscreen', True)
    window.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
    
    # The canvas will now correctly fill the entire screen
    canvas = tk.Canvas(window, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg='black')
    canvas.pack()
    
    # The dot will now start in the center of the full screen
    dot = canvas.create_oval(
        (PREVIEW_WIDTH/2) - DOT_SIZE / 2,
        (PREVIEW_HEIGHT/2) - DOT_SIZE / 2,
        (PREVIEW_WIDTH/2) + DOT_SIZE / 2,
        (PREVIEW_HEIGHT/2) + DOT_SIZE / 2,
        fill='red', 
        outline='red'
    )
    # --- END FULLSCREEN MODIFICATION ---
    
    print("GUI ready. Waiting for data...")
    
    # 5. Start the loops
    read_serial_data()
    update_prediction() # No 'model' argument
    window.mainloop()
    

except FileNotFoundError as e:
    print(f"\n--- FATAL ERROR: FILE NOT FOUND ---")
    print(f"Could not find a required file.")
    print(f"Details: {e}")
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