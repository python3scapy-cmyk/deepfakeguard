import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sr = sample_rate
        self.n_mfcc = n_mfcc
    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            return self.extract_features_from_array(y)
        except Exception as e:
            print(f"Error processing: {audio_path} -^> {e}")
            return None

    def extract_features_from_array(self, y):
        """Same 26-dim feature vector as extract_features(), but computed
        directly on an in-memory float array (already at self.sr). Used by
        the live AASISTDetector on raw microphone chunks, guaranteeing the
        runtime features match exactly what the XGBoost model was trained on."""
        y = np.asarray(y, dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta_mfcc.T, axis=0)
        return np.hstack([mfcc_mean, delta_mean])
