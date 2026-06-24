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
