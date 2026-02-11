import serial
import time
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets
import sys

# --- CONFIG ---
PORT = 'COM5'
PACKET_SIZE = 14 # 4 (sync) + 2+2+2 (sensors) + 4 (timestamp)
VCC = 3.3
SCALING = 0.3434343434343

def capture_and_align():
    ser = serial.Serial(PORT, 2000000, timeout=0.1)
    ser.reset_input_buffer()
    print("Waiting for Trigger...")

    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if "START" in line: break

    raw_bytes = bytearray()
    start = time.time()
    while time.time() - start < 5.5:
        if ser.in_waiting > 0:
            raw_bytes.extend(ser.read(ser.in_waiting))
    ser.close()

    print("Processing Data...")
    data = np.frombuffer(raw_bytes, dtype=np.uint8)
    
    # Find all 0xFFFFFFFF headers (4 consecutive 255s)
    # This is much more robust than 0xAABB
    is_255 = (data == 255)
    header_hits = is_255[:-3] & is_255[1:-2] & is_255[2:-1] & is_255[3:]
    indices = np.where(header_hits)[0]
    
    # Filter to ensure indices are spaced correctly (roughly every 14 bytes)
    # and don't overflow the buffer
    indices = indices[indices + PACKET_SIZE <= len(data)]
    
    # Fast extraction
    s1 = np.array([data[i+4:i+6].view(np.uint16)[0] for i in indices])
    s2 = np.array([data[i+6:i+8].view(np.uint16)[0] for i in indices])
    s3 = np.array([data[i+8:i+10].view(np.uint16)[0] for i in indices])
    ts = np.array([data[i+10:i+14].view(np.uint32)[0] for i in indices])

    return s1, s2, s3, ts

# Run it
s1, s2, s3, ts = capture_and_align()

# Clean Up: Remove samples where time went backwards (misalignment check)
valid_mask = np.diff(ts, prepend=ts[0]-1) > 0
s1, s2, s3, ts = s1[valid_mask], s2[valid_mask], s3[valid_mask], ts[valid_mask]

# Convert
time_sec = (ts.astype(np.float64) - ts[0]) / 1_000_000.0
s1_v = (s1 * (VCC / 1023.0)) / SCALING
s2_v = (s2 * (VCC / 1023.0)) / SCALING
s3_v = (s3 * (VCC / 1023.0)) / SCALING

# Save for later inspection
np.save("clean_capture.npy", np.column_stack((time_sec, s1_v, s2_v, s3_v)))

print(f"Captured {len(ts)} clean samples.")

# Plot
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True)
p = win.addPlot(title="Clean Aligned Data")
p.addLegend()
p.plot(time_sec, s1_v, pen='r', name='Sensor 1')
p.plot(time_sec, s2_v, pen='g', name='Sensor 2')
p.plot(time_sec, s3_v, pen='b', name='Sensor 3')
app.exec()