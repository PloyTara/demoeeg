import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import cwt, morlet2

# --- Load .mat file ---
mat_file_path = r'D:\data\Subject_foreigners_15-10-2025\Subject_file_mat\high_f\41\untitled0002.mat'
mat_data = loadmat(mat_file_path)
data = mat_data['data']
fs = 2000  # Sampling rate

# --- Select channel ---
channel_1 = data[:, 0]

# --- Define wavelet parameters ---
freqs = np.linspace(1, 60, 100)  
widths = (fs * 6) / (2 * np.pi * freqs)  

# --- Perform Continuous Wavelet Transform (CWT) ---
cwt_result = cwt(channel_1, morlet2, widths, w=6)  # w=6 = standard Morlet wavelet

# --- Plot time-frequency heatmap ---
time = np.arange(channel_1.shape[0]) / fs 

plt.figure(figsize=(14, 6))
plt.pcolormesh(time, freqs, np.abs(cwt_result), shading='auto')
plt.title('Wavelet Transform - Channel 1')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Amplitude')
plt.ylim(0, 60)
plt.tight_layout()
plt.show()
