import serial
import re
import time
import numpy as np

# --- Config ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 2000000

print("Connecting to serial port...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
print("Connected! Starting calibration data feed.")
print("\n--- INSTRUCTIONS ---")
print("1. Keep your hand far away from the box to find the 'IDLE' values.")
print("2. Touch each of the 4 side plates (Left, Right, Front, Back) firmly.")
print("3. Write down the IDLE (min) and TOUCH (max) values for each.")
print("\nPress Ctrl+C to stop.")

# This regex pattern finds all 5 channel/error pairs on a single line
pattern = re.compile(r"Channel \d+: \d+, error :(-?\d+)")

# Store the min/max values we've seen
min_vals = [0] * 5
max_vals = [-np.inf] * 5 # Start with negative infinity

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
            
            # Find all 5 error values
            matches = pattern.findall(line)
            
            if len(matches) == 5:
                try:
                    # Get all 5 error values as integers
                    errors = [int(val) for val in matches]
                    
                    # Update min/max
                    for i in range(5):
                        if errors[i] < min_vals[i]: min_vals[i] = errors[i]
                        if errors[i] > max_vals[i]: max_vals[i] = errors[i]
                    
                    # Print them on one line
                    print(f"Ch0: {errors[0]: 4d} | Ch1: {errors[1]: 4d} | Ch2: {errors[2]: 4d} | Ch3: {errors[3]: 4d} | Ch4: {errors[4]: 4d}  ", end='\r')
                    
                except ValueError:
                    pass # Ignore parse errors
        
        time.sleep(0.01) # Small delay

except KeyboardInterrupt:
    print("\n\n--- Calibration Finished ---")
    print("Your CALIBRATION_DATA should look like this (fill in your values):")
    print("CALIBRATION_DATA = {")
    print(f"    # Plate: (IDLE_VALUE, TOUCH_VALUE)")
    print(f"    'ch0': ({min_vals[0]}, {max_vals[0]}),  # Left")
    print(f"    'ch1': ({min_vals[1]}, {max_vals[1]}),  # Front")
    print(f"    'ch2': ({min_vals[2]}, {max_vals[2]}),  # (Base or unused)")
    print(f"    'ch3': ({min_vals[3]}, {max_vals[3]}),  # Right")
    print(f"    'ch4': ({min_vals[4]}, {max_vals[4]}),  # Back/Down")
    print("}")
finally:
    if ser.is_open:
        ser.close()