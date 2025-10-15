import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch

# Load .mat file
mat_file_path = r'D:\data\Subject_foreigners_15-10-2025\Subject_file_mat\high_f\41\untitled0002.mat'
mat_data = loadmat(mat_file_path)

# Load EEG data
data = mat_data['data']  # shape: [samples x channels]
fs = 2000  # sampling rate

# Select channels
channel_1 = data[:, 0]
channel_2 = data[:, 1]

# Calculate PSD using Welch's method
frequencies_1, psd_1 = welch(channel_1, fs=fs, nperseg=1024)
frequencies_2, psd_2 = welch(channel_2, fs=fs, nperseg=1024)

# Filter frequencies between 2 and 40 Hz
freq_range = (frequencies_1 >= 2) & (frequencies_1 <= 50)
frequencies_1 = frequencies_1[freq_range]
psd_1 = psd_1[freq_range]
psd_2 = psd_2[freq_range]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(frequencies_1, 10 * np.log10(psd_1), label='Channel 1')
plt.plot(frequencies_1, 10 * np.log10(psd_2), label='Channel 2')
plt.title('PSD Comparison (2–50 Hz): Channel 1 vs Channel 2')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB)')
plt.legend()
plt.grid(True)

# Set x-axis ticks every 2 Hz
plt.xticks(np.arange(2, 52, 2))

plt.tight_layout()
plt.show()
