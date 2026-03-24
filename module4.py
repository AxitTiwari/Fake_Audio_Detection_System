import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==============================
# CONFIG
# ==============================
FEATURE_PATH = "features"
MODEL_PATH = "model/model.h5"

EPOCHS = 25
BATCH_SIZE = 32

# ==============================
# GPU OPTIMIZATION
# ==============================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU enabled")
    except:
        pass

# ==============================
# LOAD DATA
# ==============================
print("\nLoading data...")

X = np.load(os.path.join(FEATURE_PATH, "X.npy"))
y = np.load(os.path.join(FEATURE_PATH, "y.npy"))

# Normalize (VERY IMPORTANT)
X = X / np.max(X)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# TF.DATA PIPELINE (FASTER)
# ==============================
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==============================
# MODEL
# ==============================
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_model(X.shape[1:])

# ==============================
# COMPILE
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),  # lower LR improves stability
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS
# ==============================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True),
    ReduceLROnPlateau(factor=0.3, patience=3, verbose=1)
]

# ==============================
# TRAIN
# ==============================
print("\nTraining model...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==============================
# EVALUATION
# ==============================
print("\nEvaluating model...")

loss, accuracy = model.evaluate(val_ds)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Predictions
y_pred_probs = model.predict(val_ds)
y_pred = (y_pred_probs > 0.5).astype("int32")

# ==============================
# REPORT
# ==============================
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
disp.plot()
plt.show()

# ==============================
# TRAINING GRAPHS
# ==============================
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.legend()
plt.show()

# ==============================
# PREDICTION FUNCTION (FIXED)
# ==============================
def predict_audio(file_path):
    print("\nPredicting:", file_path)

    audio, sr = librosa.load(file_path, sr=16000)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize safely
    if mel_db.shape[1] < 128:
        pad_width = 128 - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :128]

    # Normalize same as training
    mel_db = mel_db / np.max(mel_db)

    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    prediction = model.predict(mel_db)[0][0]

    if prediction > 0.5:
        print(f"Fake Voice (Confidence: {prediction:.2f})")
    else:
        print(f"Real Voice (Confidence: {1 - prediction:.2f})")


# ==============================
# TEST PREDICTION
# ==============================
predict_audio("processed_data/real/sample.wav")
