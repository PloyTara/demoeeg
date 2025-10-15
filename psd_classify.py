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

# --- Band power calculation ---
def band_power(frequencies, psd, fmin, fmax):
    band = (frequencies >= fmin) & (frequencies <= fmax)
    power = np.trapz(psd[band], frequencies[band])
    return power

# --- Classification function using Welch band power ---
def classify_by_power(channel_data, fs, target_freqs, bandwidth):
    results = {}
    for f in target_freqs:
        low = f - bandwidth
        high = f + bandwidth
        filtered = bandpass_filter(channel_data, low, high, fs)
        freqs, psd = welch(filtered, fs=fs, nperseg=1024)
        power = band_power(freqs, psd, low, high)
        results[f] = power
    return results

# --- Load EEG data ---
mat_file_path = r'D:\data\Subject_foreigners_15-10-2025\Subject_file_mat\high_f\41\untitled0002.mat'
mat_data = loadmat(mat_file_path)
data = mat_data['data']  # shape: [samples x channels]
fs = 2000

# --- Select EEG channels ---
channel_1 = data[:, 0]
channel_2 = data[:, 1]

# --- Compute raw PSD for reference ---
frequencies, psd_1 = welch(channel_1, fs=fs, nperseg=1024)
_, psd_2 = welch(channel_2, fs=fs, nperseg=1024)

# --- Band Power Summary ---
bands = {
    'Alpha (8–13 Hz)': (8, 13),
    'Beta (13–30 Hz)': (13, 30),
    'Gamma (30–50 Hz)': (30, 50)
}
print("Band Power Summary:")
for band_name, (fmin, fmax) in bands.items():
    p1 = band_power(frequencies, psd_1, fmin, fmax)
    p2 = band_power(frequencies, psd_2, fmin, fmax)
    print(f" - {band_name}: Channel 1 = {p1:.4e}, Channel 2 = {p2:.4e}")

# --- Frequency classification (41–45 Hz) ---
target_freqs = np.arange(41, 46, 0.2)
bandwidth = 1.0

print("\nClassifying Frequency Response (41–45 Hz):")
results_all = {}
for ch_name, ch_data in zip(["Channel 1", "Channel 2"], [channel_1, channel_2]):
    results = classify_by_power(ch_data, fs, target_freqs, bandwidth)
    best_freq = max(results, key=results.get)
    best_power = results[best_freq]
    results_all[ch_name] = results
    print(f"📡 {ch_name} responds best to {best_freq} Hz (Power = {best_power:.4e})")

# --- Plot classification result ---
plt.figure(figsize=(8, 4))
for ch_name, powers in results_all.items():
    plt.plot(target_freqs, [powers[f] for f in target_freqs], marker='o', label=ch_name)

plt.title("EEG Response (Welch Band Power) in 41–45 Hz")
plt.xlabel("Target Frequency (Hz)")
plt.ylabel("Band Power (µV²)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
