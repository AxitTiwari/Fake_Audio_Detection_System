import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm

# CONFIG
RAW_DATA_PATH = "raw_data"
CLEAN_DATA_PATH = "cleaned_data"
METADATA_PATH = "metadata/metadata.csv"

TARGET_SR = 16000
MAX_AMPLITUDE = 0.99
SILENCE_THRESHOLD = 0.01   # very important

os.makedirs(CLEAN_DATA_PATH, exist_ok=True)
os.makedirs("metadata", exist_ok=True)

# HELPER FUNCTIONS
def is_silent(audio):
    """Check if audio is silent"""
    rms = np.sqrt(np.mean(audio**2))
    return rms < SILENCE_THRESHOLD


def normalize_audio(audio):
    """Normalize peak amplitude to 0.99"""
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return (audio / max_val) * MAX_AMPLITUDE


def process_file(file_path):
    """Load and clean audio"""
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

        # Remove silent files
        if is_silent(audio):
            return None

        # Normalize
        audio = normalize_audio(audio)

        duration = len(audio) / TARGET_SR

        return audio, duration

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None



# MAIN PIPELINE
metadata = []

for label in ["real", "fake"]:
    input_folder = os.path.join(RAW_DATA_PATH, label)
    output_folder = os.path.join(CLEAN_DATA_PATH, label)

    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)

    print(f"\nProcessing {label} files...")

    for i, file in enumerate(tqdm(files)):
        file_path = os.path.join(input_folder, file)

        result = process_file(file_path)

        if result is None:
            continue

        audio, duration = result

        # Save cleaned file
        new_filename = f"{label}_{i}.wav"
        save_path = os.path.join(output_folder, new_filename)

        sf.write(save_path, audio, TARGET_SR)

        # Store metadata
        metadata.append({
            "file_path": save_path,
            "label": label,
            "duration": duration,
            "sample_rate": TARGET_SR
        })


# SAVE METADATA
df = pd.DataFrame(metadata)
df.to_csv(METADATA_PATH, index=False)

print("\n✅ Cleaning complete!")
print(f"Total processed files: {len(df)}")