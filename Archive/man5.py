import serial
import csv
import re
import numpy as np
import time
import math
import datetime
from collections import deque
import re

# --- STEP 1: IMPORT URSINA ---
from ursina import *

# --- Configuration (Unchanged) ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 2000000
NUM_CHANNELS = 5
CALIBRATION_SAMPLES = 100 

MAX_SENSOR_DIFFERENCE = 1000.0 

Z_MAX_SENSITIVITY = 5

Z_CLICK_THRESHOLD = 0.2     

# --- GUI Settings (Unchanged) ---
SEQ_LENGTH = 1
# TUNE THESE "KNOBS":
MIN_CUTOFF = 0.1  
BETA = 0.1      

# --- Global Variables ---
ser = None
raw_data_buffer = deque(maxlen=SEQ_LENGTH)
dynamic_baselines = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# --- NEW: URSINA/CAD Globals ---
gesture_state = "move"    # "move", "extrude"
current_cube = None       
last_z_state = "low"      
cursor_3d = None          

# --- NEW: Color Feature Globals ---
color_list = [color.lime, color.red, color.lime, color.red, color.yellow, color.orange, color.magenta]
current_color_index = 0

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

# --- SENSOR/CALIBRATION FUNCTIONS (Unchanged) ---
def calibrate_baselines(ser_conn, num_samples=5):
    """(Unchanged) Reads 5 baseline values (R, L, D, U, Z)"""
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
    """(Unchanged) Calculates position and returns [x, y, z_norm]"""
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
    """(Unchanged) Non-blocking read"""
    global raw_data_buffer, ser
    if not ser: return 
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
                    pass 
    except Exception as e:
        print(f"Serial read error: {e}")

# --- NEW: Helper function to get latest gesture data ---
def get_smooth_gestures():
    if not raw_data_buffer:
        return None 

    raw_data = list(raw_data_buffer)[-1] 
    current_time = time.time()
    
    raw_x, raw_y, raw_z = calculate_xy_position(raw_data)

    smooth_x = x_filter.filter(current_time, raw_x)
    smooth_y = y_filter.filter(current_time, raw_y)
    smooth_z = z_filter.filter(current_time, raw_z) 
    
    return smooth_x, smooth_y, smooth_z

# --- NEW: URSINA'S BUILT-IN 'input' FUNCTION ---
# This function is automatically called by Ursina when a key is pressed
def input(key):
    global current_color_index
    
    # <--- NEW: 'c' key cycles through our color_list
    if key == 'c':
        current_color_index = (current_color_index + 1) % len(color_list)
        # Get the simple name of the color for printing
        color_name = color_list[current_color_index].name
        print(f"\nColor changed to: {color_name}")

    if key == 'escape':
        quit()
        
# --- MODIFIED: 'update' loop ---
def update():
    global gesture_state, current_cube, last_z_state, cursor_3d, current_color_index
    
    read_serial_data()
    gestures = get_smooth_gestures()
    if not gestures:
        print("Waiting for data...", end='\r')
        return
        
    smooth_x, smooth_y, smooth_z = gestures
    
    world_x = smooth_x * 10 
    world_y = smooth_y * 10
    
    
    is_clicking = smooth_z < Z_CLICK_THRESHOLD
    z_just_clicked = False
    
    if is_clicking and last_z_state == "low":
        z_just_clicked = True
    
    last_z_state = "high" if is_clicking else "low"
    
    # 5. --- THE MAIN CAD LOGIC (STATE MACHINE) ---
    
    if gesture_state == "move":
        # <--- MODIFIED: Cursor now shows the selected color
        cursor_3d.position = (world_x, 0.01, world_y)
        cursor_3d.color = color_list[current_color_index]
        
        if z_just_clicked:
            print("STATE: STAMP (Click 1)")
            current_cube = Entity(
                model='cube',
                color=color.cyan, # "In-progress" color
                texture='white_cube',
                scale=(1, 0.1, 1), 
                position=cursor_3d.position
            )
            gesture_state = "extrude" # Switch to next state

    elif gesture_state == "extrude":
        # <--- MODIFIED: Cursor is cyan to show "extrude" mode
        cursor_3d.color = color.cyan 
        
        # --- EXTRUDE the cube ---
        new_height = 0.1 + (smooth_z * 10)
        
        if current_cube:
            current_cube.scale_y = new_height
            current_cube.y = new_height / 2 
            
        if z_just_clicked:
            print("STATE: FINALIZE (Click 2)")
            if current_cube:
                # <--- MODIFIED: Set the cube's final color
                current_cube.color = color_list[current_color_index]
                current_cube.collider = 'box'   
            
            current_cube = None
            gesture_state = "move" 
            
    print(f"X: {world_x:.2f} Y: {world_y:.2f} | Z: {smooth_z:.2f} | State: {gesture_state}", end='\r')

# --- Main Script ---
try:
    print("Starting 'Gesture CAD'...")

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01) 
    print("Connection successful!")
    
    calibrate_baselines(ser, num_samples=CALIBRATION_SAMPLES)
    
    print("Starting Ursina 3D environment...")
    app = Ursina()
    
    floor = Entity(model='plane', scale=(20, 1, 20), color=color.gray, texture='white_cube', texture_scale=(20,20))
    
    # <--- MODIFIED: Cursor starts with the first color in the list
    cursor_3d = Entity(model='quad', color=color_list[current_color_index], scale=(1, 1, 1))
    cursor_3d.rotation_x = 90 
    
    Sky()
    EditorCamera()
    
    print("GUI ready. 'Click' to stamp, 'Pull' to extrude, 'Click' again to set.")
    print("--- Press 'c' to cycle colors! ---")
    
    app.run()
    
except serial.serialutil.SerialException as e:
    print(f"\n--- FATAL ERROR ---")
    print(f"Could not open serial port '{SERIAL_PORT}'.")
    print("Please check the port and ensure you have permissions.")
    print(f"Original error: {e}")
except Exception as e:
    print(f"A fatal error occurred: {e}")
finally:
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")