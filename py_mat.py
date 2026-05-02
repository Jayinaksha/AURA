import serial
import csv
import datetime
import re # Import the regular expressions library

# --- Configuration ---
# This should be your Arduino's serial port on Linux/macOS
SERIAL_PORT = '/dev/ttyACM0'
# On Windows, it would be 'COM3', 'COM4', etc.

BAUD_RATE = 115200
OUTPUT_FILE = 'mpr121_error_data.csv'
NUM_CHANNELS = 5
# ---------------------

print(f"Attempting to connect to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")

try:
    # Initialize serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print("Connection successful! Waiting for data...")

    # Open the CSV file in write mode
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Create a new, more descriptive header for the CSV file
        # Using "Error" to match your new terminology
        header = ['Timestamp']
        for i in range(NUM_CHANNELS):
            header.extend([f'Channel_{i}_Filtered', f'Channel_{i}_Error'])
        csv_writer.writerow(header)
        print(f"Created '{OUTPUT_FILE}' with header: {header}")

        # Dictionary to store readings for one complete cycle
        current_readings = {}

        # Main loop to read and process data
        while True:
            try:
                line = ser.readline().decode('utf-8').strip()

                # If the line is empty, just continue to the next one
                if not line:
                    continue

                # --- THIS IS THE CORRECTED REGULAR EXPRESSION ---
                # It is designed to match your exact output format like:
                # "03:43:34.616 -> Channel 0: 967, error :-4"
                # It handles potential extra spaces for robustness.
                match = re.search(r"Channel\s*(\d+):\s*(\d+),\s*error\s*:(-?\d+)", line)

                if match:
                    # Extract the captured groups from the regex
                    channel_num = int(match.group(1))
                    filtered_val = int(match.group(2))
                    error_val = int(match.group(3))
                    
                    # Store both values for the current channel
                    current_readings[channel_num] = (filtered_val, error_val)

                    # Once all 5 channels are collected, write the complete row to the file
                    if len(current_readings) == NUM_CHANNELS:
                        # Get a high-precision timestamp from the computer
                        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        
                        # Build the row with both filtered and error values, in order
                        row_data = [timestamp]
                        for i in range(NUM_CHANNELS):
                            row_data.extend(current_readings[i]) # Adds (filtered, error) tuple

                        # Write to the CSV file and print to the console
                        csv_writer.writerow(row_data)
                        print(f"Logged: {row_data}")
                        
                        # Reset the dictionary for the next batch of readings
                        current_readings = {}

            except KeyboardInterrupt:
                print("\nStopping data logging. Closing files.")
                break
            except Exception as e:
                print(f"A fatal error occurred: {e}")
                break

finally:
    # Ensure the serial port is closed on exit
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
