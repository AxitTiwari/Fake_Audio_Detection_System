import numpy as np
import librosa
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
MODEL_PATH = "model/sample_model.h5"
AUDIO_PATH = "C:\\Users\\DELL\\Downloads\\test.wav"   

TARGET_SR = 16000
DURATION = 4  # seconds
SAMPLES = TARGET_SR * DURATION

model = load_model(MODEL_PATH)

def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SR)

    # Trim silence
    audio, _ = librosa.effects.trim(audio)

    # Fix length
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    return audio

def extract_features(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SR,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize (important)
    mel_db = mel_db[:, :128]

    # Normalize (VERY IMPORTANT)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # Add dimensions for model
    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    return mel_db


def predict_audio(file_path):
    audio = preprocess_audio(file_path)
    features = extract_features(audio)

    print("Feature shape:", features.shape)

    prediction = model.predict(features)[0][0]

    print("Raw prediction:", prediction)

    if prediction > 0.5:
        print(f"🎯 Fake Voice ({prediction*100:.2f}% confidence)")
    else:
        print(f"🎯 Real Voice ({(1-prediction)*100:.2f}% confidence)")


if __name__ == "__main__":
    predict_audio(AUDIO_PATH)