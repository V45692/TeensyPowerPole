import serial
import numpy as np
import time

# --- CONFIGURATION ---
PORT = 'COM5'  # <--- DOUBLE CHECK THIS in Arduino IDE (Tools > Port)
FILENAME = "pole_test_data.npy"
PACKET_SIZE = 8  # Our struct: s1(2), s2(2), s3(2), padding(2)
SAMPLE_RATE = 300000 
DURATION = 2     # Seconds to record
TOTAL_SAMPLES = SAMPLE_RATE * DURATION

def run_collector():
    try:
        # 1. Open Serial with a large buffer
        ser = serial.Serial(PORT, 2000000, timeout=1)
        ser.set_buffer_size(rx_size=10_000_000) # 10MB buffer
        print(f"Connected to {PORT}. Waiting for 'START' from Teensy...")
        print("Note: Press the button/trigger on Pin 2 of your Teensy now.")

        # 2. Wait for the START trigger
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if "START" in line:
                print(">>> START Signal Received! Recording...")
                break
        
# 3. Read the binary blob
        bytes_to_read = TOTAL_SAMPLES * PACKET_SIZE
        raw_data = ser.read(bytes_to_read)
        
        print(f"Captured {len(raw_data)} bytes.")

        # --- THE FIX ---
        # Calculate how many full 8-byte packets we actually got
        num_complete_packets = len(raw_data) // PACKET_SIZE
        # Trim the raw_data so it is a perfect multiple of 8
        trimmed_data = raw_data[:num_complete_packets * PACKET_SIZE]
        
        # Convert trimmed binary to Numpy
        data = np.frombuffer(trimmed_data, dtype=np.uint16).reshape(-1, 4)
        # ----------------
        
        # Strip the padding (the 4th column)
        clean_data = data[:, :3]
        
        # 5. Save and Finish
        np.save(FILENAME, clean_data)
        print(f"Successfully saved {num_complete_packets} samples to {FILENAME}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    run_collector()