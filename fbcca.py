import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

# --- Load .mat file ---
mat_file_path = r'D:\data\Subject_foreigners_15-10-2025\Subject_file_mat\high_f\41\untitled0002.mat'
mat_data = loadmat(mat_file_path)
data = mat_data['data']
fs = 2000  # Sampling rate

# --- Select EEG channels ---
eeg = data[:, [0, 1]]  # ใช้ channel 1 และ 2

# --- สร้าง Reference Signals ---
def generate_reference_signals(freq, n_samples, fs, n_harmonics=2):
    t = np.arange(n_samples) / fs
    refs = []
    for i in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * freq * i * t))
        refs.append(np.cos(2 * np.pi * freq * i * t))
    return np.stack(refs, axis=1)

# --- สร้าง Filter Bank ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# --- FBCCA Parameters ---
test_freqs = np.arange(41, 46)  # 41–45 Hz
n_banks = 5  # จำนวน filter bank
harmonics = 2
n_samples = eeg.shape[0]

# --- Filter bank cutoff ranges (log-spaced)
cutoff_bands = [
    (8 / (i + 1), 88 / (i + 1)) for i in range(n_banks)
]

# --- FBCCA Processing ---
fbcca_results = []

for freq in test_freqs:
    corr_sum = 0
    for i, (lowcut, highcut) in enumerate(cutoff_bands):
        # 1. Filter EEG
        eeg_filt = bandpass_filter(eeg, lowcut, highcut, fs)

        # 2. Generate reference
        ref = generate_reference_signals(freq, n_samples, fs, harmonics)

        # 3. Apply CCA
        cca = CCA(n_components=1)
        cca.fit(eeg_filt, ref)
        U, V = cca.transform(eeg_filt, ref)
        corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

        # 4. Weight by (1 / bank index)
        corr_sum += (corr ** 2) / (i + 1)
    
    fbcca_results.append(corr_sum)

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.plot(test_freqs, fbcca_results, marker='o')
plt.title('FBCCA Score vs Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('FBCCA Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Best Frequency ---
best_freq = test_freqs[np.argmax(fbcca_results)]
print(f"🎯 Highest FBCCA score at: {best_freq} Hz (score = {max(fbcca_results):.4f})")
