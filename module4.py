import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==============================
# CONFIG
# ==============================
FEATURE_PATH = "features"
MODEL_PATH = "model/model.h5"

EPOCHS = 25
BATCH_SIZE = 32

# ==============================
# LOAD DATA
# ==============================
print("\nLoading data...")

X = np.load(os.path.join(FEATURE_PATH, "X.npy"))
y = np.load(os.path.join(FEATURE_PATH, "y.npy"))

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==============================
# TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# BUILD MODEL
# ==============================
def build_model(input_shape):
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Block 2
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Block 3
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Flatten
    model.add(Flatten())

    # Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output
    model.add(Dense(1, activation='sigmoid'))

    return model


model = build_model(X.shape[1:])

# ==============================
# COMPILE
# ==============================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS (IMPORTANT)
# ==============================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

# ==============================
# TRAIN
# ==============================
print("\nTraining model...")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks
)

# ==============================
# SAVE FINAL MODEL
# ==============================
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)

print("\n✅ Model saved at:", MODEL_PATH)

# ==============================
# EVALUATION
# ==============================
print("\nEvaluating model...")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Predictions
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype("int32")

# ==============================
# CLASSIFICATION REPORT
# ==============================
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ==============================
# TRAINING GRAPH
# ==============================
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.legend()
plt.show()

# ==============================
# PREDICTION FUNCTION
# ==============================
import librosa

def predict_audio(file_path):
    print("\nPredicting:", file_path)

    audio, sr = librosa.load(file_path, sr=16000)

    # Create mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize (IMPORTANT)
    mel_db = mel_db[:, :128]  # adjust if needed

    # Add dimensions
    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    prediction = model.predict(mel_db)[0][0]

    if prediction > 0.5:
        print(f"Fake Voice (Confidence: {prediction:.2f})")
    else:
        print(f"Real Voice (Confidence: {1 - prediction:.2f})")


# ==============================
# TEST PREDICTION (OPTIONAL)
# ==============================
# predict_audio("processed_data/real/sample.wav")