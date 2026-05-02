import serial
import csv
import re
import numpy as np
import tensorflow as tf
import time
import math
import datetime
import tkinter as tk
from collections import deque
from feature_engineering import SpatialPlateFeatureEngineer, RobustFeatureEngineer
from models.trajectory_only_model import SpatialAttentionLayer
import joblib
import pandas as pd
from collections import deque
from pathlib import Path
import sys


# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200
MODEL_FILE = 'trajectory_model.h5'
NUM_CHANNELS = 5

# --- Model & GUI Settings ---
# These MUST match your model's training!
SEQ_LENGTH = 500  # Input shape (500, 48)
NUM_FEATURES = 46 # Input shape (500, 48)

# The 10 raw features we get from the Arduino
NUM_RAW_FEATURES = NUM_CHANNELS * 2 # (Filtered, Error) * 5

# GUI Window
PREVIEW_WIDTH = 600
PREVIEW_HEIGHT = 400
DOT_SIZE = 10

# --- Global Variables ---
ser = None
window = None
canvas = None
dot = None
# Use a deque to automatically keep the last 500 readings
raw_data_buffer = deque(maxlen=SEQ_LENGTH)
feature_engineer = None


def preprocess_data(raw_sequence_list):
    """
    This function now uses your loaded RobustFeatureEngineer to
    convert the (500, 12) data into (500, 46) features.
    """
    global feature_engineer # This is the fitted engineer we loaded
    
    # 1. Convert our list of 12-feature rows into a (1, 500, 12) numpy array
    #    This shape (batch_size, timesteps, features) is what process_features expects
    raw_sequence_np = np.array(raw_sequence_list).reshape(1, SEQ_LENGTH, 12) 
    
    # 2. Call the fitted processor's 'process_features' method
    #    We pass 'fit=False' to use the loaded scalers.
    #    This will handle all 46 feature calculations and normalization.
    processed_sequence, _ = feature_engineer.process_features(raw_sequence_np, fit=False, verbose=False)
    
    # 3. Return the (500, 46) feature array
    #    process_features returns (1, 500, 46), so we take the first item
    return processed_sequence[0]

def read_serial_data():
    """
    Non-blocking function to read from serial and add to the buffer.
    """
    global raw_data_buffer, ser
    
    # Temporary dict to store readings for one cycle
    current_readings = {}
    
    try:
        # Read as many lines as are waiting, to clear the serial buffer
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue

            match = re.search(r"Channel\s*(\d+):\s*(\d+),\s*error\s*:(-?\d+)", line)
            
            if match:
                channel_num = int(match.group(1))
                filtered_val = int(match.group(2))
                error_val = int(match.group(3))
                
                # Store both values
                current_readings[channel_num] = (filtered_val, error_val)

                # If we've collected all 5 channels, add them to the buffer
                if len(current_readings) == NUM_CHANNELS:
                    row_data = []
                    for i in range(NUM_CHANNELS):
                        row_data.extend(current_readings[i]) # Adds (f, e) tuple
                    
                    # Add the new 10-feature row to our buffer
                    # The deque automatically removes the oldest item
                    row_data.extend([0.0, 0.0])
                    raw_data_buffer.append(row_data)
                    
                    # Reset for next cycle
                    current_readings = {}
                    
    except Exception as e:
        print(f"Serial read error: {e}")

    # Reschedule this function to run again
    window.after(5, read_serial_data) # Check for new data every 5ms


def update_prediction(model):
    """
    This function runs ~30 times per second.
    It takes the data buffer, preprocesses it, predicts, and moves the dot.
    """
    global raw_data_buffer, canvas, dot
    
    # Wait until the buffer is full before we start predicting
    if len(raw_data_buffer) < SEQ_LENGTH:
        # Schedule the next update
        window.after(33, update_prediction, model) # ~30 FPS
        return

    # 1. Get the raw data sequence (shape will be 500x10)
    raw_sequence = list(raw_data_buffer)
    
    # 2. Preprocess the data (YOU MUST IMPLEMENT THIS)
    # This must turn the (500, 10) list into a (500, 46) numpy array
    processed_sequence = preprocess_data(raw_sequence)
    
    # 3. Reshape for the model
    # Model expects (batch_size, timesteps, features)
    model_input = processed_sequence.reshape(1, SEQ_LENGTH, NUM_FEATURES)
    
    # 4. Predict!
    # The model outputs a sequence of (500, 2) coordinates
    prediction_sequence = model.predict(model_input)
    
    # 5. Get the *last* (most recent) (x, y) coordinate from the prediction
    # Output shape is (1, 500, 2), so we take [0][-1]
    predicted_xy = prediction_sequence[0][-1] # This will be like [0.123, -0.456]
    
    # 6. Scale the output
    # The model's 'tanh' activation [cite: 34] outputs from -1 to 1.
    # We need to scale this to our window coordinates (0 to PREVIEW_WIDTH/HEIGHT).
    x_pos = (predicted_xy[0] + 1) / 2 * PREVIEW_WIDTH
    y_pos = (predicted_xy[1] + 1) / 2 * PREVIEW_HEIGHT
    
    # 7. Move the dot on the canvas
    canvas.coords(
        dot,
        x_pos - DOT_SIZE / 2,
        y_pos - DOT_SIZE / 2,
        x_pos + DOT_SIZE / 2,
        y_pos + DOT_SIZE / 2
    )
    
    # 8. Schedule this function to run again
    window.after(33, update_prediction, model) # ~30 FPS


# --- Main Script ---
try:
    # 1. Load the Keras Model
    print(f"Loading model '{MODEL_FILE}'...")
    # This may require a 'custom_objects' argument if you have custom layers
    # like 'SpatialAttentionLayer'. Let's try without it first.
    try:
        model = tf.keras.models.load_model(
            MODEL_FILE,
            custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer})
    except ValueError as e:
        print(f"Error: Could not load model. It might have custom layers.")
        print(f"Original error: {e}")
        print("You may need to define your custom layers in this script, e.g.:")
        print("model = tf.keras.models.load_model(MODEL_FILE, custom_objects={'SpatialAttentionLayer': SpatialAttentionLayer})")
        exit()
        
    model.summary()
    print("Model loaded successfully.")

    print("Loading scalers 'scalers.pkl'...")
    # Make sure this path is correct!
    scalers = joblib.load('scalers.pkl') 
    feature_engineer = scalers['feature_engineer']
    print("Scalers loaded.")

    # 2. Connect to Serial Port
    print(f"Attempting to connect to port '{SERIAL_PORT}' at {BAUD_RATE} bps.")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    print("Connection successful!")
    
    # 3. Set up the GUI
    print("Opening preview window...")
    window = tk.Tk()
    window.title("Live Trajectory Demo")
    window.geometry(f"{PREVIEW_WIDTH}x{PREVIEW_HEIGHT}")
    
    canvas = tk.Canvas(window, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT, bg='black')
    canvas.pack()
    
    # Create the 'dummy mouse' dot at the center
    dot = canvas.create_oval(
        (PREVIEW_WIDTH/2) - DOT_SIZE / 2,
        (PREVIEW_HEIGHT/2) - DOT_SIZE / 2,
        (PREVIEW_WIDTH/2) + DOT_SIZE / 2,
        (PREVIEW_HEIGHT/2) + DOT_SIZE / 2,
        fill='red', 
        outline='red'
    )
    
    print("GUI ready. Waiting for data buffer to fill...")
    
    # 4. Start the loops
    # Start reading serial data in the background
    read_serial_data()
    # Start the prediction/GUI update loop
    update_prediction(model)
    
    # 5. Start the main GUI loop
    window.mainloop()
    

except KeyboardInterrupt:
    print("\nStopping demo.")
except Exception as e:
    print(f"A fatal error occurred: {e}")
finally:
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
    if window:
        window.destroy()
        print("Window closed.")