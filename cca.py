import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt

# --- Load .mat file ---
mat_file_path = r'D:\data\Subject_foreigners_15-10-2025\Subject_file_mat\high_f\41\untitled0002.mat'
mat_data = loadmat(mat_file_path)
data = mat_data['data']
fs = 2000

# --- Function to generate reference signals ---
def generate_reference_signals(freq, n_samples, fs, n_harmonics=2):
    t = np.arange(n_samples) / fs
    refs = []
    for i in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * freq * i * t))
        refs.append(np.cos(2 * np.pi * freq * i * t))
    return np.stack(refs, axis=1)

# --- Frequencies to test ---
test_freqs = np.arange(41, 46)
n_samples = data.shape[0]

# --- CCA for each channel ---
correlations = {}  # Store correlations for each channel

for ch in [0, 1]:
    eeg = data[:, ch].reshape(-1, 1)  # one channel
    corrs = []
    for freq in test_freqs:
        ref = generate_reference_signals(freq, n_samples, fs)
        cca = CCA(n_components=1)
        cca.fit(eeg, ref)
        U, V = cca.transform(eeg, ref)
        corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        corrs.append(corr)
    correlations[f'Channel {ch+1}'] = corrs

# --- Plotting ---
plt.figure(figsize=(8, 4))
for label, corr_values in correlations.items():
    plt.plot(test_freqs, corr_values, marker='o', label=label)

plt.title('CCA Correlation vs Frequency (per Channel)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Canonical Correlation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Print best frequency for each channel ---
print("\n📊 Highest Correlation per Channel:")
for label, corr_values in correlations.items():
    best_idx = np.argmax(corr_values)
    best_freq = test_freqs[best_idx]
    best_corr = corr_values[best_idx]
    print(f" - {label}: Best response at {best_freq} Hz (corr = {best_corr:.4f})")
