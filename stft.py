import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import stft, butter, filtfilt

# --- Bandpass Filter Function ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- Load .mat file ---
mat_file_path = r'D:\data\Subject_foreigners_15-10-2025\Subject_file_mat\high_f\41\untitled0002.mat'
mat_data = loadmat(mat_file_path)
data = mat_data['data']
fs = 2000  # Sampling rate

# --- Select channels ---
channel_1 = data[:, 0]
channel_2 = data[:, 1]

# --- Apply Bandpass Filter (41–45 Hz) ---
channel_1_filt = bandpass_filter(channel_1, 41, 45, fs)
channel_2_filt = bandpass_filter(channel_2, 41, 45, fs)

# --- STFT Parameters ---
nperseg = 2048
noverlap = 1024

# --- Compute STFT ---
f1, t1, Zxx1 = stft(channel_1_filt, fs=fs, nperseg=nperseg, noverlap=noverlap)
f2, t2, Zxx2 = stft(channel_2_filt, fs=fs, nperseg=nperseg, noverlap=noverlap)

# --- Limit frequency range to 40–46 Hz ---
freq_mask = (f1 >= 40) & (f1 <= 46)  # f1 == f2

# --- Plot both channels together ---
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Channel 1
pcm1 = axs[0].pcolormesh(t1, f1[freq_mask], 10 * np.log10(np.abs(Zxx1[freq_mask])**2), shading='gouraud')
axs[0].set_title('STFT Spectrogram (Filtered 41–45 Hz) - Channel 1')
axs[0].set_ylabel('Frequency (Hz)')
fig.colorbar(pcm1, ax=axs[0], label='Power (dB)')

# Channel 2
pcm2 = axs[1].pcolormesh(t2, f2[freq_mask], 10 * np.log10(np.abs(Zxx2[freq_mask])**2), shading='gouraud')
axs[1].set_title('STFT Spectrogram (Filtered 41–45 Hz) - Channel 2')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Frequency (Hz)')
fig.colorbar(pcm2, ax=axs[1], label='Power (dB)')

plt.tight_layout()
plt.show()
