import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sr = sample_rate
        self.n_mfcc = n_mfcc
    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta_mean = np.mean(delta_mfcc.T, axis=0)
            return np.hstack([mfcc_mean, delta_mean])
        except Exception as e:
            print(f"Error processing: {audio_path} -^> {e}")
            return None

    def extract_features_from_array(self, y, sr=None):
        try:
            sr = sr or self.sr
            y = np.asarray(y, dtype=np.float32)
            if y.size == 0 or np.allclose(y, 0):
                return None
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            delta_mean = np.mean(librosa.feature.delta(mfcc).T, axis=0)
            return np.hstack([mfcc_mean, delta_mean])
        except Exception as e:
            print(f"[WARN] array feature extraction failed: {e}")
            return None

    def extract_features_from_array(self, y, sr=None):
        try:
            sr = sr or self.sr
            y = np.asarray(y, dtype=np.float32)
            if y.size == 0 or np.allclose(y, 0):
                return None
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            delta_mean = np.mean(librosa.feature.delta(mfcc).T, axis=0)
            return np.hstack([mfcc_mean, delta_mean])
        except Exception as e:
            print(f"[WARN] array feature extraction failed: {e}")
            return None
