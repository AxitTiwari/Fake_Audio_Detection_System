import os
import librosa
import numpy as np
from tqdm import tqdm

# CONFIG
DATA_PATH = "processed_data"

SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048


# STORAGE
X = []
y = []

label_map = {
    "real": 0,
    "fake": 1
}

# FEATURE FUNCTION
def extract_mel_spectrogram(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        # Convert to log scale (VERY IMPORTANT)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db

    except Exception as e:
        print(f"Error: {file_path} -> {e}")
        return None


# MAIN LOOP
for label in ["real", "fake"]:
    folder = os.path.join(DATA_PATH, label)
    files = os.listdir(folder)

    print(f"\nProcessing {label}...")

    for file in tqdm(files):
        file_path = os.path.join(folder, file)

        feature = extract_mel_spectrogram(file_path)

        if feature is None:
            continue

        X.append(feature)
        y.append(label_map[label])


# CONVERT TO NUMPY
X = np.array(X)
y = np.array(y)

# Add channel dimension (for CNN)
X = X[..., np.newaxis]


# SAVE
os.makedirs("features", exist_ok=True)

np.save("features/X.npy", X)
np.save("features/y.npy", y)

print("\n✅ Feature extraction complete!")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")