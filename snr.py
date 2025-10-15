import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
import matplotlib.pyplot as plt

# --- Load EEG .mat file ---
mat_file_path = r'D:\data\Subject_foreigners_15-10-2025\Subject_file_mat\high_f\41\untitled0002.mat'
mat_data = loadmat(mat_file_path)
data = mat_data['data']
fs = 2000  # Sampling rate

# --- Select EEG channel ---
channel = data[:, 0]

# --- calculate Power Spectral Density (PSD) byย Welch method ---
frequencies, psd = welch(channel, fs=fs, nperseg=2048)

# --- SNR calculation ที่ 41 Hz ---
target_freq = 41
delta = 1  # window ±1 Hz

# find index near 41 Hz
idx_target = np.argmin(np.abs(frequencies - target_freq))
signal_power = psd[idx_target]

# Noise power
noise_mask = ((frequencies >= target_freq - delta) & 
              (frequencies <= target_freq + delta) & 
              (frequencies != frequencies[idx_target]))
noise_power = np.mean(psd[noise_mask])

# SNR calculation (dB)
snr_db = 10 * np.log10(signal_power / noise_power)
print(f"SNR at {target_freq} Hz = {snr_db:.2f} dB")

# --- Plot PSD and SNR window ---
plt.figure(figsize=(10, 5))
plt.semilogy(frequencies, psd, label='PSD')
plt.axvline(x=target_freq, color='red', linestyle='--', label='Target Freq')
plt.axvspan(target_freq - delta, target_freq + delta, color='gray', alpha=0.2, label='Noise Window')
plt.title(f'PSD with SNR window around {target_freq} Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (V²/Hz)')
plt.xlim(35,50)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

