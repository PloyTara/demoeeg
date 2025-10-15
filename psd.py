import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch, butter, filtfilt

# --- Bandpass Filter Function ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- Function to compute band power ---
def band_power(frequencies, psd, fmin, fmax):
    band = (frequencies >= fmin) & (frequencies <= fmax)
    power = np.trapz(psd[band], frequencies[band])
    return power

# --- Load .mat file ---
mat_file_path = r'D:\data\Subject_foreigners_15-10-2025\Subject_file_mat\high_f\41\untitled0002.mat'
mat_data = loadmat(mat_file_path)
data = mat_data['data']  # EEG data [samples x channels]
fs = 2000  # Sampling rate

# --- Select channels ---
channel_1 = data[:, 0]
channel_2 = data[:, 1]

# --- Calculate PSD (before filtering) ---
frequencies, psd_1 = welch(channel_1, fs=fs, nperseg=1024)
_, psd_2 = welch(channel_2, fs=fs, nperseg=1024)

# --- Band power analysis ---
bands = {
    'Alpha (8-13 Hz)': (8, 13),
    'Beta (13-30 Hz)': (13, 30),
    'Gamma (30-50 Hz)': (30, 50)
}

print("Band Power (µV²/Hz):")
for band_name, (fmin, fmax) in bands.items():
    power_1 = band_power(frequencies, psd_1, fmin, fmax)
    power_2 = band_power(frequencies, psd_2, fmin, fmax)
    print(f"{band_name}:  Channel 1 = {power_1:.4e},  Channel 2 = {power_2:.4e}")

# --- Apply Bandpass Filter around 41 Hz ---
channel_1_filtered = bandpass_filter(channel_1, 40, 42, fs)
channel_2_filtered = bandpass_filter(channel_2, 40, 42, fs)

# --- Calculate PSD after filtering ---
frequencies_filt, psd_1_filt = welch(channel_1_filtered, fs=fs, nperseg=1024)
_, psd_2_filt = welch(channel_2_filtered, fs=fs, nperseg=1024)

# --- Plot PSD (full 2–50 Hz) ---
freq_range = (frequencies >= 2) & (frequencies <= 50)
plt.figure(figsize=(12, 6))
plt.plot(frequencies[freq_range], 10 * np.log10(psd_1[freq_range]), label='Channel 1')
plt.plot(frequencies[freq_range], 10 * np.log10(psd_2[freq_range]), label='Channel 2')
plt.title('PSD Comparison (2–50 Hz): Channel 1 vs Channel 2')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB)')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(2, 52, 2))
plt.tight_layout()
plt.show()

# --- Plot PSD After Bandpass Filter (40–42 Hz) ---
focus_range = (frequencies_filt >= 35) & (frequencies_filt <= 45)
plt.figure(figsize=(10, 5))
plt.plot(frequencies_filt[focus_range], 10 * np.log10(psd_1_filt[focus_range]), label='Filtered Ch1 (40–42Hz)')
plt.plot(frequencies_filt[focus_range], 10 * np.log10(psd_2_filt[focus_range]), label='Filtered Ch2 (40–42Hz)')
plt.title('Bandpass Filtered PSD (40–42 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
