import serial
import csv
import re
import pyautogui
import time
import math
import datetime
import tkinter as tk

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'  # Linux/macOS
# SERIAL_PORT = 'COM3'           # Windows (Example)
BAUD_RATE = 2000000
OUTPUT_FILE = 'mpr121_user_pattern_data.csv' # Using your new filename
NUM_CHANNELS = 5

# --- Your Pattern Settings ---
# How many times to loop over the whole pattern
ROUNDS_OF_COLLECTION = 1
# How long to wait between points (controls speed of the pattern)
# Smaller = faster.
STEP_DELAY = 0.01 # 20 milliseconds

# ---------------------

def get_circle_trajectory(screen_width, screen_height, steps=200):
    """
    Generates a list of (x, y) coordinates for a circle pattern.
    """
    trajectory = []
    center_x = screen_width // 2
    center_y = screen_height // 2
    radius_x = screen_width // 4
    radius_y = screen_height // 4 # Use 1/4 of screen as radius

    for i in range(steps):
        angle = (i / steps) * 2 * math.pi
        x = int(center_x + radius_x * math.cos(angle))
        y = int(center_y + radius_y * math.sin(angle))
        trajectory.append((x, y))
    print(f"Generated circle trajectory with {steps} points.")
    return trajectory

def get_line_trajectory(screen_width, screen_height, steps=100):
    """
    Generates a list of (x, y) coordinates for a back-and-forth line.
    """
    trajectory = []
    start_x = screen_width // 4
    end_x = screen_width - (screen_width // 4)
    y = screen_height // 2
    
    # Line from left to right
    for i in range(steps):
        x = int(start_x + (end_x - start_x) * (i / steps))
        trajectory.append((x, y))
        
    # Line from right to left
    for i in range(steps):
        x = int(end_x - (end_x - start_x) * (i / steps))
        trajectory.append((x, y))
        
    print(f"Generated line trajectory with {steps * 2} points.")
    return trajectory

def preview_trajectory(trajectory, screen_width, screen_height, step_delay):
    """
    Shows the pattern in a new, dedicated 'dummy' window.
    This is "the curser will follow a pattern"
    """
    print("\n--- STEP 1: PREVIEW ---")
    print("Watch the new window. It will show you the pattern one time.")

    # --- Settings for the preview window ---
    PREVIEW_WIDTH = 600  # Width of the pop-up window
    PREVIEW_HEIGHT = 400 # Height of the pop-up window
    DOT_SIZE = 10        # Size of the red dot
    # ----------------------------------------

    try:
        # 1. Create the main window
        window = tk.Tk()
        window.title("Pattern Preview")
        window.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")

        # 2. Create a black canvas to draw on
        canvas = tk.Canvas(window, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg='black')
        canvas.pack()

        # 3. Create the 'dummy mouse' (a red dot)
        # We place it at the start of the scaled trajectory
        start_x, start_y = trajectory[0]
        
        # --- Scale from screen coordinates to window coordinates ---
        scaled_start_x = (start_x / screen_width) * PREVIEW_WIDTH
        scaled_start_y = (start_y / screen_height) * PREVIEW_HEIGHT

        dot = canvas.create_oval(
            scaled_start_x - DOT_SIZE / 2,
            scaled_start_y - DOT_SIZE / 2,
            scaled_start_x + DOT_SIZE / 2,
            scaled_start_y + DOT_SIZE / 2,
            fill='red', 
            outline='red'
        )
        
        # Give the window a moment to appear
        window.update()
        time.sleep(0.5)

        # 4. Loop through the pattern
        for x, y in trajectory:
            # Scale the full-screen (x, y) to our small window
            scaled_x = (x / screen_width) * PREVIEW_WIDTH
            scaled_y = (y / screen_height) * PREVIEW_HEIGHT
            
            # Move the red dot
            canvas.coords(
                dot,
                scaled_x - DOT_SIZE / 2,
                scaled_y - DOT_SIZE / 2,
                scaled_x + DOT_SIZE / 2,
                scaled_y + DOT_SIZE / 2
            )
            
            # Refresh the window and pause
            window.update()
            time.sleep(step_delay) # Use the *exact* same speed as the data collection
        
        # 5. Pause at the end, then close
        time.sleep(1)
        window.destroy()

    except Exception as e:
        print(f"  > Warning: Could not show preview window. Error: {e}")
        print("  > Please try to follow the pattern anyway.")
    
    print("...Preview complete.")

def run_collection(ser, csv_writer, trajectory):
    """
    This runs your "finger will do the same pattern" step.
    This version is CORRECTED to read the new, single-line serial format.
    """
    
    # This regex pattern finds all 5 channel/error pairs on a single line
    pattern = re.compile(r"Channel \d+: (\d+), error :(-?\d+)")
    
    print("--- Starting data collection loop. Follow the pattern! ---")
    
    # Flush the serial buffer before we start
    # to get rid of any old, stale data.
    ser.reset_input_buffer()
    
    for target_x, target_y in trajectory:
        loop_start_time = time.time()
        
        # --- NEW, CORRECTED DATA READING LOGIC ---
        
        row_data = []
        found_data = False
        
        # 1. We will read new lines until we find one that matches
        #    our 5-channel format.
        #    We set a timeout so we don't get stuck if serial is slow.
        read_timeout = loop_start_time + (STEP_DELAY * 0.9) # 90% of our step time
        
        while time.time() < read_timeout:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue
                
                # 2. Use findall to get ALL 5 matches from this ONE line
                matches = pattern.findall(line)
                
                # 3. Check if we found exactly 5 channels
                if len(matches) == 5:
                    try:
                        # 4. Build the row for the CSV
                        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        row_data = [timestamp]
                        
                        for match in matches:
                            row_data.append(float(match[0])) # Filtered value
                            row_data.append(float(match[1])) # Error value
                        
                        # 5. Add the target mouse coordinates
                        row_data.extend([target_x, target_y])
                        
                        found_data = True
                        break # Got our data, exit this 'while' loop

                    except ValueError:
                        print(f"Serial parse error on line: {line}")
                        # Loop continues to try and find a good line
            else:
                # Small pause to not hog the CPU while waiting
                time.sleep(0.001)
        
        # --- END OF NEW LOGIC ---

        # 6. Write data (even if we found nothing)
        if found_data:
            csv_writer.writerow(row_data)
        else:
            # THIS IS CRITICAL: If we missed a reading, we write a
            # blank row (with 0s) to keep the trajectory data
            # perfectly aligned in time.
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            blank_sensor_data = [0.0] * (NUM_CHANNELS * 2) # 10 zeros
            row_data = [timestamp] + blank_sensor_data + [target_x, target_y]
            csv_writer.writerow(row_data)

        # 7. Wait for the *remainder* of the step delay
        # This ensures each step takes EXACTLY STEP_DELAY seconds,
        # making the pattern speed consistent and followable.
        elapsed_time = time.time() - loop_start_time
        sleep_duration = STEP_DELAY - elapsed_time
        
        if sleep_duration > 0:
            time.sleep(sleep_duration)

# --- Main Script ---
print("This script will record data using your 'Show-and-Repeat' method.")
print("1. The script will move the mouse to show you a pattern.")
print("2. The script will wait 3 seconds.")
print("3. You will then do the pattern with your finger.")
print("4. The script will record your finger and the pattern's (x,y) data.")
print("Make sure to install 'pyautogui': pip install pyautogui")

try:
    print(f"\nAttempting to connect to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01) # Short timeout
    print("Connection successful!")
    
    # Let user choose pattern
    print("\nSelect a pattern to map:")
    print("  1: Circle")
    print("  2: Horizontal Line")
    choice = input("Enter 1 or 2: ")
    
    screen_width, screen_height = pyautogui.size()
    
    if choice == '1':
        trajectory = get_circle_trajectory(screen_width, screen_height, steps=200)
    elif choice == '2':
        trajectory = get_line_trajectory(screen_width, screen_height, steps=100)
    else:
        print("Invalid choice. Exiting.")
        exit()

    # STEP 1: Show the user the pattern first
    preview_trajectory(trajectory, screen_width, screen_height, STEP_DELAY)
    
    print(f"\nWe will now record this pattern {ROUNDS_OF_COLLECTION} times.")
    print("When I say 'GO', start moving your finger in the pattern.")
    print(f"You must match the pattern's speed (total time: {len(trajectory) * STEP_DELAY:.1f} sec)")
    print("\nPress Ctrl+C to stop at any time.")

    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # --- THIS BLOCK IS MODIFIED TO MATCH SCRIPT 1'S HEADER ---
        # Build the header
        header = ['Timestamp'] # <-- ADDED
        for i in range(NUM_CHANNELS):
            # <-- MODIFIED: Add both Filtered and Error columns
            header.extend([f'Channel_{i}_Filtered', f'Channel_{i}_Error'])
        
        header.extend(['Mouse_X', 'Mouse_Y']) # <-- KEPT FROM SCRIPT 2
        
        csv_writer.writerow(header)
        print(f"\nCreated '{OUTPUT_FILE}' with header: {header}")
        # --- END OF MODIFIED BLOCK ---

        # STEP 2: Wait 3 seconds
        print("\n--- STEP 2: GET READY ---")
        print("Get ready to start moving your finger...")
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        
        print("\n--- STEP 3: GO! START MOVING! ---")
        
        # Start the collection loops
        for round_num in range(1, ROUNDS_OF_COLLECTION + 1):
            print(f"--- Starting Round {round_num} / {ROUNDS_OF_COLLECTION} ---")
            
            # Run the pattern (this records data, mouse does not move)
            run_collection(ser, csv_writer, trajectory)
            
            print(f"--- Completed Round {round_num} ---")
            # Small pause between rounds
            time.sleep(0.5)
            if round_num < ROUNDS_OF_COLLECTION:
                print("Get ready for the next round...")
                time.sleep(1.5)

        print("\n\n--- DATA COLLECTION COMPLETE! ---")
        print(f"Your training file '{OUTPUT_FILE}' is ready.")

except KeyboardInterrupt:
    print("\nStopping data logging. Closing files.")
except Exception as e:
    print(f"A fatal error occurred: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")