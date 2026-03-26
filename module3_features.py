import os
import librosa
import numpy as np
from tqdm import tqdm


class Config:
    DATA_PATH = "processed_data"
    OUTPUT_PATH = "features"

    SAMPLE_RATE = 16000
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048

    LABEL_MAP = {
        "real": 0,
        "fake": 1
    }


class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.X = []
        self.y = []

    def extract_mel_spectrogram(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.config.SAMPLE_RATE)

            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH,
                n_mels=self.config.N_MELS
            )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            return mel_spec_db

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def process_folder(self, label):
        folder_path = os.path.join(self.config.DATA_PATH, label)

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return

        files = os.listdir(folder_path)

        print(f"\n📂 Processing '{label}' data...")

        for file in tqdm(files):
            file_path = os.path.join(folder_path, file)

            feature = self.extract_mel_spectrogram(file_path)

            if feature is None:
                continue

            self.X.append(feature)
            self.y.append(self.config.LABEL_MAP[label])

    def run(self):
        for label in self.config.LABEL_MAP.keys():
            self.process_folder(label)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        self.X = self.X[..., np.newaxis]

        print("\n📊 Feature Extraction Summary:")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")

    def save(self):
        os.makedirs(self.config.OUTPUT_PATH, exist_ok=True)

        np.save(os.path.join(self.config.OUTPUT_PATH, "X.npy"), self.X)
        np.save(os.path.join(self.config.OUTPUT_PATH, "y.npy"), self.y)

        print("\n✅ Features saved successfully!")


def main():
    print("🚀 Starting Feature Extraction Pipeline...")

    config = Config()
    extractor = FeatureExtractor(config)

    extractor.run()
    extractor.save()

    print("\n🎉 Feature extraction completed successfully!")


if __name__ == "__main__":
    main()