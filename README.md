# 🎙️ AI Deepfake Voice Detection System

<p align="center">
  <b>Detect Real vs AI-Generated Voices using Deep Learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Flask-WebApp-green?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/Status-Working-yellow?style=for-the-badge">
</p>

---

## 🚀 Overview

This project is a **Deep Learning-based Audio Forensics System** that detects whether a voice recording is:

* ✅ **Real (Human Speech)**
* ❌ **Fake (AI-generated / Deepfake Speech)**

It combines **signal processing + neural networks + web deployment** into a complete pipeline.

---

## 🎯 Key Features

✨ Deep Learning-based classification (CNN)
🎧 Audio upload and real-time prediction
📊 Confidence score visualization
🎬 Animated modern UI (Glassmorphism design)
⚡ End-to-end pipeline (Data → Model → Web App)

---

## 🧠 System Architecture

```text id="arch001"
User Audio Input
        ↓
Preprocessing (Resample, Trim, Normalize)
        ↓
Feature Extraction (Mel Spectrogram)
        ↓
CNN Model
        ↓
Prediction (Real / Fake)
        ↓
Web Interface Output
```

---

## 🧩 Modules

### 🔹 Module 1: Data Collection & Cleaning

* Remove corrupted/silent audio
* Normalize amplitude
* Convert to `.wav`

### 🔹 Module 2: Preprocessing

* Fixed sample rate (16kHz)
* Silence trimming
* Padding/cutting to fixed length

### 🔹 Module 3: Feature Extraction

* Mel Spectrogram generation
* Log scaling
* Dataset creation (`X.npy`, `y.npy`)

### 🔹 Module 4: Model Training

* CNN-based classifier
* Binary classification (Real vs Fake)
* Evaluation metrics (Accuracy, Precision, Recall, F1)

---

## 📸 Screenshots

> 📌 Add your project screenshots here for better presentation

```
Example:
- UI Screenshot
- Prediction Result
- Spectrogram Visualization
```

---

## 💻 Tech Stack

| Category         | Technology            |
| ---------------- | --------------------- |
| Language         | Python                |
| Deep Learning    | TensorFlow / Keras    |
| Audio Processing | Librosa               |
| Data Handling    | NumPy, Pandas         |
| Web Framework    | Flask                 |
| Frontend         | HTML, CSS, JavaScript |

---

## 📁 Project Structure

```text id="struct001"
project/
├── raw_data/
├── cleaned_data/
├── processed_data/
├── features/
│   ├── X.npy
│   ├── y.npy
├── model/
│   └── model.h5
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   ├── script.js
├── app.py
├── module1_cleaning.py
├── module2_preprocessing.py
├── module3_features.py
├── module4_model.py
├── requirements.txt
```

---

## ⚙️ Installation

```bash id="inst001"
git clone <your-repo-link>
cd project
```

### Create virtual environment

```bash id="inst002"
python -m venv venv
venv\Scripts\activate
```

### Install dependencies

```bash id="inst003"
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1: Train Model

```bash id="run001"
python module1_cleaning.py
python module2_preprocessing.py
python module3_features.py
python module4_model.py
```

### Step 2: Run Web App

```bash id="run002"
python app.py
```

### Step 3: Open Browser

```text id="run003"
http://127.0.0.1:5000
```

---

## 📊 Model Performance

| Metric   | Value           |
| -------- | --------------- |
| Accuracy | ~85–95%         |
| Model    | CNN             |
| Input    | Mel Spectrogram |

> ⚠️ Performance depends on dataset size and quality

---

## ⚠️ Limitations

* Requires sufficient dataset for good generalization
* Sensitive to noisy audio
* May not detect all advanced AI voices

---

## 🚀 Future Improvements

* 🎤 Real-time microphone detection
* 📊 Spectrogram visualization in UI
* 🤖 CNN + LSTM hybrid model
* 🌐 Cloud deployment (Render / AWS)
* 🔍 Multi-class classification

---

## 🧠 Learning Outcomes

* Audio signal processing
* Deep learning for speech
* Feature engineering (Mel Spectrogram)
* End-to-end ML pipeline
* Full-stack AI application

---

## 👨‍💻 Team

* Developed by a team of 4 (B.Tech CSE)

---

## 📌 Conclusion

This project demonstrates how **Deep Learning can be applied to detect AI-generated voices**, solving real-world problems in **security, media, and digital authenticity**.

---

<p align="center">
⭐ If you like this project, give it a star!
</p>
