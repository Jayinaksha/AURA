import serial
import csv
import re
import pyautogui
import time
import datetime
import tkinter as tk
import numpy as np

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'  # Linux/macOS
# SERIAL_PORT = 'COM3'           # Windows (Example)
BAUD_RATE = 2000000
OUTPUT_FILE = 'mpr121_static_grid_data01.csv' # NEW filename
NUM_CHANNELS = 5

# --- Your NEW Grid Settings ---
GRID_SIZE_X = 15  # How many points horizontally
GRID_SIZE_Y = 15  # How many points vertically
SAMPLES_PER_POINT = 25 # How many readings to average for each point
TIME_BETWEEN_READINGS = 0.01 # Should match your Arduino's print_delay
# ------------------------------

# This regex pattern finds all 5 channel/error pairs on a single line
# This is from your script and matches your Arduino output perfectly.
data_pattern = re.compile(r"Channel \d+: (\d+), error :(-?\d+)")

def setup_preview_window():
    """
    Creates and returns the Tkinter window and canvas elements.
    """
    print("Opening preview window...")
    window = tk.Tk()
    window.title("Data Collection Grid")
    
    # Make the window a bit smaller than the screen
    screen_w, screen_h = pyautogui.size()
    win_w = int(screen_w * 0.7)
    win_h = int(screen_h * 0.7)
    
    # Center the window
    win_x = (screen_w // 2) - (win_w // 2)
    win_y = (screen_h // 2) - (win_h // 2)
    window.geometry(f"{win_w}x{win_h}+{win_x}+{win_y}")

    window.attributes('-topmost', True)
    
    canvas = tk.Canvas(window, width=win_w, height=win_h, bg='black')
    canvas.pack(fill="both", expand=True)
    
    # Create the target dot
    dot = canvas.create_oval(0, 0, 0, 0, fill='red', outline='red')
    
    window.update()
    return window, canvas, dot, win_w, win_h

def get_grid_trajectory(grid_x, grid_y, window_w, window_h):
    """
    Generates a list of (x, y) coordinates for the grid.
    These are the coordinates *inside the preview window*.
    """
    trajectory = []
    # We use a 10% margin on each side
    x_margin = window_w * 0.1
    y_margin = window_h * 0.1
    
    usable_w = window_w - (2 * x_margin)
    usable_h = window_h - (2 * y_margin)

    for iy in range(grid_y):
        # Calculate y-coordinate
        # (iy / (grid_y - 1)) gives a value from 0.0 to 1.0
        y = int(y_margin + (iy / (grid_y - 1)) * usable_h)
        
        # This makes the pattern go back-and-forth (serpentine)
        # which is faster for you to follow.
        x_range = range(grid_x)
        if iy % 2 != 0: # If on an odd row, reverse the x-direction
            x_range = reversed(x_range)
            
        for ix in x_range:
            # Calculate x-coordinate
            x = int(x_margin + (ix / (grid_x - 1)) * usable_w)
            
            # We save the *grid* coordinate (e.g., 0,0) and the
            # *screen* coordinate (e.g., 250, 150)
            trajectory.append(((ix, iy), (x, y)))
            
    print(f"Generated {len(trajectory)} grid points.")
    return trajectory

def move_dot(canvas, dot, x, y, dot_size=20):
    """
    Moves the red dot to the new target (x, y) position.
    """
    canvas.coords(
        dot,
        x - dot_size / 2,
        y - dot_size / 2,
        x + dot_size / 2,
        y + dot_size / 2
    )
    canvas.update()

def collect_avg_data(ser, samples_to_take, time_between):
    """
    Reads N samples from the Arduino and averages them.
    """
    all_readings = [] # Will store N readings
    
    ser.reset_input_buffer()
    
    print(f"  > Recording {samples_to_take} samples... HOLD STILL.")
    
    while len(all_readings) < samples_to_take:
        loop_start = time.time()
        
        line = ser.readline().decode('utf-8').strip()
        if not line:
            # Add a tiny sleep if no data, but continue loop
            time.sleep(0.001)
            continue
            
        matches = data_pattern.findall(line)
        
        if len(matches) == NUM_CHANNELS:
            try:
                current_sample = [] # Stores 10 values (5x filt, 5x err)
                for match in matches:
                    current_sample.append(float(match[0])) # Filtered
                    current_sample.append(float(match[1])) # Error
                all_readings.append(current_sample)
                
            except ValueError:
                print(f"  > Parse error on line: {line}")
                continue
        
        # Wait for the correct time delay
        elapsed = time.time() - loop_start
        if elapsed < time_between:
            time.sleep(time_between - elapsed)
            
    # Now we have all samples, let's average them
    # np.mean(all_readings, axis=0) averages vertically
    # This turns 25 rows of 10 values into 1 row of 10 values
    avg_values = np.mean(all_readings, axis=0)
    print("  > Done.")
    return avg_values


# --- Main Script ---
print("This script will record data for a STATIC GRID.")
print("It is designed for your Phase 2 'Touchless Trackpad'.")
print("---")
print("1. A window will open with a red dot.")
print("2. Move your hand to match the dot's position.")
print("3. Click THIS TERMINAL, then press [Enter] to confirm.")
print("4. HOLD STILL while the script records data for that point.")
print("5. The dot will move. Repeat.")
print("---")

try:
    print(f"Connecting to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    print("Connection successful!")
    
    # 1. Setup the visual window
    window, canvas, dot, win_w, win_h = setup_preview_window()
    
    # 2. Generate the list of points to visit
    trajectory = get_grid_trajectory(GRID_SIZE_X, GRID_SIZE_Y, win_w, win_h)
    
    # 3. Open the CSV file and write the header
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Build the header
        header = ['Timestamp']
        for i in range(NUM_CHANNELS):
            header.extend([f'Channel_{i}_Filtered_Avg', f'Channel_{i}_Error_Avg'])
        
        # We save the GRID coordinates (0-14) as the ML target
        header.extend(['Target_Grid_X', 'Target_Grid_Y'])
        
        csv_writer.writerow(header)
        print(f"Created '{OUTPUT_FILE}' with header: {header}")
        
        print("\n--- Starting Data Collection ---")
        print("Click the terminal window (not the red one) to start.")
        
        total_points = len(trajectory)
        for i, (grid_coords, screen_coords) in enumerate(trajectory):
            
            grid_x, grid_y = grid_coords
            screen_x, screen_y = screen_coords
            
            # 4. Move the dot to the next position
            move_dot(canvas, dot, screen_x, screen_y)
            
            # 5. Wait for user to be ready
            input(f"Point {i+1}/{total_points} (Grid: {grid_x},{grid_y}). Move to dot and press [Enter]...")
            
            # 6. Collect N samples and average them
            avg_data = collect_avg_data(ser, SAMPLES_PER_POINT, TIME_BETWEEN_READINGS)
            
            # 7. Write the single, averaged row to the CSV
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            row_data = [timestamp] + list(avg_data) + [grid_x, grid_y]
            csv_writer.writerow(row_data)

        print("\n\n--- STATIC GRID COLLECTION COMPLETE! ---")
        print(f"Your training file '{OUTPUT_FILE}' is ready.")
        
        # Close the preview window
        window.destroy()

except KeyboardInterrupt:
    print("\nStopping data logging. Closing files.")
except Exception as e:
    print(f"A fatal error occurred: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
    if 'window' in locals():
        try:
            window.destroy()
        except:
            pass