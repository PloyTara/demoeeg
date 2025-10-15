import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch

# --- Load EEG ---
mat_file_path = r'D:\data\Subject_foreigners_15-10-2025\Subject_file_mat\high_f\41\untitled0002.mat'
mat_data = loadmat(mat_file_path)
data = mat_data['data']
fs = 2000
channel = data[:, 0]  # Channel 1

# --- Sliding window parameters ---
window_size = 1.0    # seconds (ละเอียดขึ้น)
step_size = 0.25     # seconds
nperseg = 2048       # ความละเอียดความถี่สูงขึ้น

samples_per_win = int(window_size * fs)
samples_step = int(step_size * fs)
n_windows = int((len(channel) - samples_per_win) / samples_step) + 1

# --- Frequency range to show ---
freq_min = 2
freq_max = 50
target_freq = 41

# --- Dynamic PSD calculation ---
psd_matrix = []
time_points = []

for i in range(n_windows):
    start = i * samples_step
    end = start + samples_per_win
    segment = channel[start:end]
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
    
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    psd_matrix.append(10 * np.log10(psd[freq_mask]))
    time_points.append(start / fs)

freqs_plot = freqs[freq_mask]
psd_matrix = np.array(psd_matrix).T  # shape: [freqs x time]

# --- Plot dynamic PSD ---
plt.figure(figsize=(14, 6))
plt.pcolormesh(time_points, freqs_plot, psd_matrix, shading='gouraud', cmap='hot')
plt.axhline(y=target_freq, color='red', linestyle='--', label=f'{target_freq} Hz')
plt.title('Dynamic PSD (Sliding Window) with Target Frequency at 41 Hz')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power (dB)')
plt.legend()
plt.tight_layout()
plt.show()
