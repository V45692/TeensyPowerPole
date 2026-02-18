import numpy as np
import matplotlib.pyplot as plt

data = np.load("pole_test_data.npy")

# Plot the middle of the file where the hit likely is
# Or plot the whole thing if it's only 1 second
plt.figure(figsize=(12, 6))
plt.plot(data[:, 0], label="Impact (S1)", alpha=0.7)
plt.plot(data[:, 2], label="Opposite (S3)", alpha=0.7)
plt.title("Full 1-Second Capture at 300kHz")
plt.xlabel("Sample Number")
plt.ylabel("ADC Value (0-1023)")
plt.legend()
plt.show()