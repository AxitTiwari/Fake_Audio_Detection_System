import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# CONFIG
INPUT_PATH = "cleaned_data"
OUTPUT_PATH = "processed_data"

TARGET_SR = 16000  # seconds
TARGET_DURATION = 4 
TARGET_LENGTH = TARGET_SR * TARGET_DURATION

os.makedirs(OUTPUT_PATH, exist_ok=True)

# FUNCTIONS
def trim_silence(audio):
    """Remove silence using librosa"""
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)
    return trimmed_audio


def pad_or_cut(audio):
    """Fix audio length"""
    if len(audio) > TARGET_LENGTH:
        # CUT
        return audio[:TARGET_LENGTH]
    else:
        # PAD
        padding = TARGET_LENGTH - len(audio)
        return np.pad(audio, (0, padding), mode='constant')


def process_audio(file_path):
    """Full preprocessing pipeline"""
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

        # Trim silence
        audio = trim_silence(audio)

        # Fix length
        audio = pad_or_cut(audio)

        return audio

    except Exception as e:
        print(f"Error: {file_path} -> {e}")
        return None



# MAIN PIPELINE
for label in ["real", "fake"]:
    input_folder = os.path.join(INPUT_PATH, label)
    output_folder = os.path.join(OUTPUT_PATH, label)

    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)

    print(f"\nProcessing {label} files...")

    for i, file in enumerate(tqdm(files)):
        file_path = os.path.join(input_folder, file)

        processed_audio = process_audio(file_path)

        if processed_audio is None:
            continue

        save_name = f"{label}_{i}.wav"
        save_path = os.path.join(output_folder, save_name)

        sf.write(save_path, processed_audio, TARGET_SR)


if __name__ == "__main__":
    print("\n✅ Preprocessing Complete!")