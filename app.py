from flask import Flask, render_template, request
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "model/sample_model.h5"
model = load_model(MODEL_PATH)

TARGET_SR = 16000
TARGET_DURATION = 4
TARGET_LENGTH = TARGET_SR * TARGET_DURATION


def preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SR)

    # Trim silence
    audio, _ = librosa.effects.trim(audio)

    # Fix length
    if len(audio) > TARGET_LENGTH:
        audio = audio[:TARGET_LENGTH]
    else:
        audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))

    return audio

def extract_features(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=TARGET_SR)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to match training shape
    mel_db = mel_db[:, :128]

    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    return mel_db


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Copy file to static for playback
    static_path = os.path.join("static", file.filename)
    os.makedirs("static", exist_ok=True)

    with open(filepath, "rb") as f:
        with open(static_path, "wb") as f2:
            f2.write(f.read())

    # Pipeline
    audio = preprocess_audio(filepath)
    features = extract_features(audio)

    prediction = model.predict(features)[0][0]

    if prediction > 0.5:
        result = "Fake Voice"
        confidence = round(prediction * 100, 2)
    else:
        result = "Real Voice"
        confidence = round((1 - prediction) * 100, 2)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        filename=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)