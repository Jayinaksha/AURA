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

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 2000000
NUM_CHANNELS = 5
CALIBRATION_SAMPLES = 100 # <--- NEW: Number of samples to average for baseline
# --- Smoothing Configuration ---
EMA_ALPHA = 0.5 # <--- TUNE THIS: 0.1=Very Smooth (laggy), 0.9=Very Responsive (jittery)
smooth_x = 0.0
smooth_y = 0.0
# --- kp_x and kp_y are REMOVED ---
# We now output normalized [-1, 1] values
MAX_SENSOR_DIFFERENCE = 1000.0 # This is no longer used, but left here

# --- GUI Settings ---
SEQ_LENGTH = 1
PREVIEW_WIDTH = 800
PREVIEW_HEIGHT = 600
DOT_SIZE = 10

# --- Global Variables ---
ser = None
window = None
canvas = None
dot = None
raw_data_buffer = deque(maxlen=SEQ_LENGTH)
# --- IDLE_BASELINE is REMOVED ---
# This will store [right_baseline, left_baseline, down_baseline, up_baseline]
dynamic_baselines = np.array([0.0, 0.0, 0.0, 0.0]) # <--- NEW
# TUNE THESE "KNOBS":
MIN_CUTOFF = 0.1  # How much to smooth jitter (e.g., 1.0)
BETA = 0.1      # How responsive to be to fast moves (e.g., 0.007)


# --- ONE-EURO FILTER CLASS (PASTE THIS) ---
# This class implements the 1-Euro Filter
# Based on the paper by Gery Casiez et al.
# http://www.lifl.fr/~casiez/1e/
# ---------------------------------------------
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0):
        """Initialize the filter."""
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def _alpha(self, cutoff):
        """Compute the smoothing factor alpha."""
        te = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + te)

    def filter(self, t, x):
        """Filter the noisy signal x at time t."""
        t_e = t - self.t_prev
        
        # --- Handle rare case of no time elapsed ---
        if t_e <= 1e-6:
            self.t_prev = t
            return self.x_prev

        # --- The filter logic ---
        # 1. Calculate rate of change (derivative)
        dx = (x - self.x_prev) / t_e
        
        # 2. Smooth the derivative
        alpha_d = self._alpha(1.0) # Derivative cutoff (fixed at 1Hz)
        self.dx_prev = (alpha_d * dx) + (1.0 - alpha_d) * self.dx_prev
        
        # 3. Calculate dynamic cutoff for the *signal*
        # This is the "magic" part:
        # Cutoff frequency increases with speed (beta * |derivative|)
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        
        # 4. Smooth the signal
        alpha = self._alpha(cutoff)
        self.x_prev = (alpha * x) + (1.0 - alpha) * self.x_prev
        
        # 5. Update time
        self.t_prev = t
        
        return self.x_prev
# --- END OF FILTER CLASS ---
initial_time = time.time()
x_filter = OneEuroFilter(initial_time, 0.0, min_cutoff=MIN_CUTOFF, beta=BETA)
y_filter = OneEuroFilter(initial_time, 0.0, min_cutoff=MIN_CUTOFF, beta=BETA)

def calibrate_baselines(ser_conn, num_samples=5):
    """
    <--- NEW FUNCTION ---
    Reads the first few samples from the sensor to set a dynamic baseline.
    """
    global dynamic_baselines
    print("--- CALIBRATING ---")
    print(f"Please keep hands away from the sensor for {num_samples} readings...")
    samples = []
    
    while len(samples) < num_samples:
        try:
            line = ser_conn.readline().decode('utf-8').strip()
            if not line:
                continue
            matches = re.findall(r"Channel \d+: (\d+), error :(-?\d+)", line)
            
            if len(matches) == 5:
                row_data = []
                for match in matches:
                    row_data.append(float(match[0]))
                    row_data.append(float(match[1]))
                
                # Extract the 4 error values IN THE ORDER they are used below
                # right_val = raw_data[7]
                # left_val  = raw_data[1]
                # lower_val = raw_data[9]
                # upper_val = raw_data[3]
                baseline_frame = [
                    row_data[7], # right
                    row_data[1], # left
                    row_data[9], # down
                    row_data[3]  # up
                ]
                samples.append(baseline_frame)
                print(f"Collected sample {len(samples)}/{num_samples}...")

        except Exception as e:
            print(f"Error during calibration read: {e}. Skipping line.")
            time.sleep(0.01) # Avoid spamming errors
    
    # Calculate the average for each channel
    dynamic_baselines = np.mean(samples, axis=0)
    print("--- CALIBRATION COMPLETE ---")
    print(f"Baselines set to (R, L, D, U): {dynamic_baselines}")
    
    # Clear any data that built up in the serial buffer during calibration
    ser_conn.reset_input_buffer()
    time.sleep(0.1) # Give it a moment to settle

def calculate_xy_position(raw_data):
    """
    Calculates position using a "Center of Mass" approach.
    NOW uses dynamic baselines and outputs normalized [-1, 1] values.
    """
    global dynamic_baselines # <--- MODIFIED: Use global baselines
    
    # raw_data is a list of 10 values:
    # [ch0_f, ch0_e, ch1_f, ch1_e, ch2_f, ch2_e, ch3_f, ch3_e, ch4_f, ch4_e]
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    
    # Your mapping (still looks jumbled, but using it as requested)
    right_val = raw_data[7] # ch3_e
    left_val  = raw_data[1] # ch0_e
    lower_val = raw_data[9] # ch4_e
    upper_val = raw_data[3] # ch1_e

    try:
        # --- 1. Make all signals positive ---
        # <--- MODIFIED: Use dynamic baselines ---
        # We assume error gets *more negative* as hand is closer
        r_sig = max(0, dynamic_baselines[0] - right_val) # right
        l_sig = max(0, dynamic_baselines[1] - left_val)  # left
        d_sig = max(0, dynamic_baselines[2] - lower_val) # down
        u_sig = max(0, dynamic_baselines[3] - upper_val) # up
        
        # --- 2. "Center of Mass" Logic (Unchanged) ---
        total_signal = r_sig + l_sig + u_sig + d_sig
        
        if total_signal < 1e-6:
            return [0.0, 0.0]

        # 3. Calculate weighted average for X (Unchanged)
        x_norm = (r_sig * -1.0 + l_sig * 1.0) / total_signal
        
        # 4. Calculate weighted average for Y (Unchanged)
        y_norm = (d_sig * -1.0 + u_sig * 1.0) / total_signal

        # This "un-squishes" the coordinate space
        
        # --- 5. Apply Sensitivity and Deadzone ---
        # <--- MODIFIED: Removed kp_x and kp_y ---
        x_out = x_norm
        y_out = y_norm 

        # Your original deadzone logic (still works on [-1, 1] range)
        if(x_out <= 0.075 and x_out >= -0.075): x_out = 0
        if(y_out <= 0.10 and y_out >= -0.10): y_out = 0
        
        return [x_out, y_out]

    except IndexError:
        return [0.0, 0.0] # Buffer was empty
    except Exception as e:
        print(f"Error in calculate_xy: {e}")
        return [0.0, 0.0]

def read_serial_data():
    """
    (This function is unchanged)
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

                    raw_data_buffer.append(row_data) # Add the [10 values] list
                    
                except ValueError: 
                    print(f"Serial parse error: Could not convert values in line: {line}")
    except Exception as e:
        print(f"Serial read error: {e}")

    if window:
        window.after(5, read_serial_data)


def update_prediction():
    """
    Runs periodically, calculates position,
    and applies the OneEuroFilter for smoothing.
    """
    global raw_data_buffer, canvas, dot
    global x_filter, y_filter # <-- Get the filters

    if not raw_data_buffer:
        print(f"Waiting for first serial reading...", end='\r')
        if window:
            window.after(33, update_prediction)
        return

    raw_sequence = list(raw_data_buffer) 
    raw_data = raw_sequence[0]
    
    # --- NEW: Get current time for the filter ---
    current_time = time.time()
    
    # 1. Get the "raw" predicted position
    predicted_xy = calculate_xy_position(raw_data)
    raw_x = predicted_xy[0]
    raw_y = predicted_xy[1]

    # 2. --- NEW: Apply OneEuroFilter ---
    # This replaces the simple EMA logic
    smooth_x = x_filter.filter(current_time, raw_x)
    smooth_y = y_filter.filter(current_time, raw_y)
    # --- End of NEW filter logic ---

    # 3. Scale the *smoothed* [-1, 1] output to screen
    x_pos = (smooth_x + 1) / 2 * PREVIEW_WIDTH
    y_pos = (smooth_y + 1) / 2 * PREVIEW_HEIGHT
    
    print(f"Norm (X, Y): [{smooth_x:.3f}, {smooth_y:.3f}]  |  Screen (X, Y): [{x_pos:.1f}, {y_pos:.1f}]", end='\r')
    
    # 4. Move the dot
    if canvas:
        canvas.coords(
            dot,
            x_pos - DOT_SIZE / 2, y_pos - DOT_SIZE / 2,
            x_pos + DOT_SIZE / 2, y_pos + DOT_SIZE / 2
        )
    
    if window:
        window.after(33, update_prediction)

# --- Main Script ---
try:
    print("Starting direct-map.")

    # 3. Connect to Serial Port
    print(f"Attempting to connect to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")
    # Set a timeout so calibration doesn't hang forever if no data
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) 
    print("Connection successful!")
    
    # --- 3b. CALIBRATE BASELINES (NEW STEP) ---
    calibrate_baselines(ser, num_samples=CALIBRATION_SAMPLES)
    
    # 4. Set up the GUI
    print("Opening preview window...")
    window = tk.Tk()
    window.title("Live Trajectory Demo (Direct Map + Calibrated)")

    PREVIEW_WIDTH = window.winfo_screenwidth()
    PREVIEW_HEIGHT = window.winfo_screenheight()
    
    print(f"Detected screen size: {PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
    window.attributes('-fullscreen', True)
    window.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
    
    canvas = tk.Canvas(window, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg='black')
    canvas.pack()
    
    dot = canvas.create_oval(
        (PREVIEW_WIDTH/2) - DOT_SIZE / 2, (PREVIEW_HEIGHT/2) - DOT_SIZE / 2,
        (PREVIEW_WIDTH/2) + DOT_SIZE / 2, (PREVIEW_HEIGHT/2) + DOT_SIZE / 2,
        fill='red', outline='red'
    )
    
    print("GUI ready. Waiting for data...")
    
    # 5. Start the loops
    read_serial_data()
    update_prediction()
    window.mainloop()
    

except FileNotFoundError as e:
    print(f"\n--- FATAL ERROR: FILE NOT FOUND ---")
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
            pass