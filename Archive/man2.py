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
CALIBRATION_SAMPLES = 100 

MAX_SENSOR_DIFFERENCE = 1000.0 
Z_MAX_SENSITIVITY = 5.0 # <--- TUNE THIS! (e.g., 12000.0)
Z_CLICK_THRESHOLD = 0.4     # <--- TUNE THIS! (0.0 to 1.0)

# --- GUI Settings ---
SEQ_LENGTH = 1
PREVIEW_WIDTH = 800
PREVIEW_HEIGHT = 600
DOT_SIZE = 10
drawing_color = "cyan" # <--- NEW: Color for drawing

# --- Global Variables ---
ser = None
window = None
canvas = None
dot = None
raw_data_buffer = deque(maxlen=SEQ_LENGTH)
dynamic_baselines = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
last_pos = None # <--- NEW: To store the last (x, y) for drawing
# TUNE THESE "KNOBS":
MIN_CUTOFF = 0.1  
BETA = 0.1      

# --- ONE-EURO FILTER CLASS (Unchanged) ---
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def _alpha(self, cutoff):
        te = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + te)

    def filter(self, t, x):
        t_e = t - self.t_prev
        
        if t_e <= 1e-6:
            self.t_prev = t
            return self.x_prev

        dx = (x - self.x_prev) / t_e
        alpha_d = self._alpha(1.0)
        self.dx_prev = (alpha_d * dx) + (1.0 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        alpha = self._alpha(cutoff)
        self.x_prev = (alpha * x) + (1.0 - alpha) * self.x_prev
        self.t_prev = t
        return self.x_prev
# --- END OF FILTER CLASS ---
initial_time = time.time()
x_filter = OneEuroFilter(initial_time, 0.0, min_cutoff=MIN_CUTOFF, beta=BETA)
y_filter = OneEuroFilter(initial_time, 0.0, min_cutoff=MIN_CUTOFF, beta=BETA)
z_filter = OneEuroFilter(initial_time, 0.0, min_cutoff=MIN_CUTOFF, beta=BETA)

def calibrate_baselines(ser_conn, num_samples=5):
    """
    (Unchanged from last version)
    Reads 5 baseline values (R, L, D, U, Z)
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
                
                baseline_frame = [
                    row_data[7], row_data[1], row_data[9], row_data[3], row_data[5]
                ]
                samples.append(baseline_frame)
                print(f"Collected sample {len(samples)}/{num_samples}...")

        except Exception as e:
            print(f"Error during calibration read: {e}. Skipping line.")
            time.sleep(0.01)
    
    dynamic_baselines = np.mean(samples, axis=0)
    print("--- CALIBRATION COMPLETE ---")
    print(f"Baselines set to (R, L, D, U, Z): {dynamic_baselines}")
    
    ser_conn.reset_input_buffer()
    time.sleep(0.1)

def calculate_xy_position(raw_data):
    """
    (Unchanged from last version)
    Calculates position and returns [x, y, z_norm]
    """
    global dynamic_baselines 
    
    right_val = raw_data[7] 
    left_val  = raw_data[1] 
    lower_val = raw_data[9] 
    upper_val = raw_data[3] 
    z_val     = raw_data[5] 

    try:
        r_sig = max(0, dynamic_baselines[0] - right_val) 
        l_sig = max(0, dynamic_baselines[1] - left_val)  
        d_sig = max(0, dynamic_baselines[2] - lower_val) 
        u_sig = max(0, dynamic_baselines[3] - upper_val) 
        z_sig = max(0, dynamic_baselines[4] - z_val)     
        
        total_signal = r_sig + l_sig + u_sig + d_sig
        
        x_norm = 0.0
        y_norm = 0.0

        if total_signal > 1e-6:
            x_norm = (r_sig * -1.0 + l_sig * 1.0) / total_signal
            y_norm = (d_sig * -1.0 + u_sig * 1.0) / total_signal
        
        z_norm = max(0.0, min(1.0, z_sig / Z_MAX_SENSITIVITY))
        
        x_out = x_norm
        y_out = y_norm 
        
        if(x_out <= 0.075 and x_out >= -0.075): x_out = 0
        if(y_out <= 0.10 and y_out >= -0.10): y_out = 0
        
        return [x_out, y_out, z_norm]

    except IndexError:
        return [0.0, 0.0, 0.0] 
    except Exception as e:
        print(f"Error in calculate_xy: {e}")
        return [0.0, 0.0, 0.0]

def read_serial_data():
    """
    (Unchanged from last version)
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
                    
                except ValueError: 
                    print(f"Serial parse error: Could not convert values in line: {line}")
    except Exception as e:
        print(f"Serial read error: {e}")

    if window:
        window.after(5, read_serial_data)


def update_prediction():
    """
    <--- MODIFIED: Now draws lines when Z is active ---
    """
    global raw_data_buffer, canvas, dot
    global x_filter, y_filter, z_filter
    global last_pos # <--- NEW

    if not raw_data_buffer:
        print(f"Waiting for first serial reading...", end='\r')
        if window:
            window.after(33, update_prediction)
        return

    raw_sequence = list(raw_data_buffer) 
    raw_data = raw_sequence[0]
    
    current_time = time.time()
    
    # 1. Get raw X, Y, Z
    predicted_xyz = calculate_xy_position(raw_data)
    raw_x = predicted_xyz[0]
    raw_y = predicted_xyz[1]
    raw_z = predicted_xyz[2]

    # 2. Apply filters
    smooth_x = x_filter.filter(current_time, raw_x)
    smooth_y = y_filter.filter(current_time, raw_y)
    smooth_z = z_filter.filter(current_time, raw_z) 
    
    # 3. Scale X/Y to screen
    x_pos = (smooth_x + 1) / 2 * PREVIEW_WIDTH
    y_pos = (smooth_y + 1) / 2 * PREVIEW_HEIGHT
    current_pos = (x_pos, y_pos) # <--- NEW: Store as tuple
    
    # --- 4. MODIFIED: Handle Z-axis "drawing" logic ---
    click_state = "IDLE"
    if smooth_z < Z_CLICK_THRESHOLD:
        # Hand is close (in Z), "click" is active
        canvas.itemconfig(dot, fill=drawing_color, outline='white')
        click_state = "DRAWING"

        # --- DRAW THE LINE ---
        if last_pos is not None:
            canvas.create_line(
                last_pos[0], last_pos[1],     # From point
                current_pos[0], current_pos[1], # To point
                fill=drawing_color, 
                width=DOT_SIZE,
                tags="drawing" # <--- NEW: Add tag to clear later
            )
        # --- END OF DRAW LOGIC ---
        
    else:
        # Hand is away, "idle"
        canvas.itemconfig(dot, fill='red', outline='red')
    
    # 5. Update print statement
    print(f"Norm (X,Y,Z): [{smooth_x:.2f}, {smooth_y:.2f}, {smooth_z:.2f}] | State: {click_state} (Press 'c' to clear)", end='\r')
    
    # 6. Move the dot (unchanged)
    if canvas:
        canvas.coords(
            dot,
            x_pos - DOT_SIZE / 2, y_pos - DOT_SIZE / 2,
            x_pos + DOT_SIZE / 2, y_pos + DOT_SIZE / 2
        )
    
    # --- 7. NEW: Update last_pos ---
    last_pos = current_pos # Always update last_pos for the *next* frame
    
    if window:
        window.after(33, update_prediction)

def clear_canvas(event):
    """
    <--- NEW FUNCTION ---
    Deletes all canvas items tagged with "drawing".
    """
    global canvas
    canvas.delete("drawing")
    print("\nCanvas cleared!")

# --- Main Script ---
try:
    print("Starting 3D-Draw.")

    print(f"Attempting to connect to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) 
    print("Connection successful!")
    
    calibrate_baselines(ser, num_samples=CALIBRATION_SAMPLES)
    
    print("Opening preview window...")
    window = tk.Tk()
    window.title("Live Trajectory Demo (3D Draw)")

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
    
    # --- NEW: Bind the 'c' key to the clear_canvas function ---
    window.bind('<c>', clear_canvas)
    print("GUI ready. Press 'c' to clear drawings.")
    
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