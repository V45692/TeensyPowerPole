import numpy as np
import pandas as pd

# Load your captured data
data = np.load("pole_test_data.npy")

# Create a DataFrame with headers
df = pd.DataFrame(data, columns=['Sensor_1', 'Sensor_2', 'Sensor_3'])

# Export to CSV
df.to_csv("pole_test_data.csv", index=False)

print("Done! You can now open 'pole_test_data.csv' in Excel.")