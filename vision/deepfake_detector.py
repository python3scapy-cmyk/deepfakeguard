"""
Week 3 visual deepfake classifier.

Implements the cascaded-inference design from the roadmap:
  Stage 1 (every scored frame): cheap Laplacian texture check.
  Stage 2 (only if Stage 1 is ambiguous): real deepfake classifier
           inference, if the model could be loaded; otherwise falls
           back to a mock score so the rest of the pipeline (fusion,
           dashboard contract) can still be developed/demoed.

Stage 2 model: prithivMLmods/deepfake-detector-model-v1 (SigLIP2-based,
fine-tuned for binary deepfake image classification, ~92.9M params,
reports ~94.4% accuracy on its own held-out test set). This replaces
the originally-planned aaronchong888/DeepFake-Detect repo, which turned
out to be a "train it yourself" pipeline with no downloadable pretrained
checkpoint -- not usable on our timeline. This HF model auto-downloads
its weights on first run (needs internet once; cached locally after).

Frame subsampling (every Nth frame) + a rolling window average are
both implemented, matching the Day 3/Day 4 spec.
"""
import os
import time
from collections import deque

import cv2
import numpy as np

DEFAULT_MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
# This model's label space: 0 = fake, 1 = real (confirmed on the model card).
FAKE_LABEL_INDEX = 0
REAL_LABEL_INDEX = 1


class DeepfakeDetector:
    def __init__(self, model_path=None, device="cpu", hf_model_name=DEFAULT_MODEL_NAME):
        self.device = device
        self.frame_counter = 0
        self.subsample = 3
        self.window = deque(maxlen=30)
        self.model = None
        self.processor = None
        self.backend = None  # "siglip_hf", "efficientnet_local", or None (mock)
        bundled = os.path.join(getattr(cv2.data, "haarcascades", ""), "haarcascade_frontalface_default.xml")
        self.cascade_path = bundled if os.path.exists(bundled) else "haarcascade_frontalface_default.xml"

        # Priority 1: a local custom checkpoint explicitly passed in (e.g. if
        # someone later trains/exports their own EfficientNet weights).
        if model_path and os.path.exists(model_path):
            try:
                import torch
                from efficientnet_pytorch import EfficientNet
                self.model = EfficientNet.from_name("efficientnet-b0", num_classes=1)
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.to(device)
                self.model.eval()
                self.backend = "efficientnet_local"
                print(f"[INFO] Loaded local EfficientNet checkpoint from {model_path}")
                return
            except Exception as e:
                print(f"[WARNING] Could not load local EfficientNet checkpoint: {e}")
                self.model = None

        # Priority 2: real pretrained deepfake classifier from Hugging Face
        # (auto-downloads on first run, cached under ~/.cache/huggingface after).
        try:
            import torch
            from transformers import AutoImageProcessor, SiglipForImageClassification
            print(f"[INFO] Loading deepfake classifier '{hf_model_name}' (first run downloads ~370MB)...")
            self.processor = AutoImageProcessor.from_pretrained(hf_model_name)
            self.model = SiglipForImageClassification.from_pretrained(hf_model_name)
            self.model.to(device)
            self.model.eval()
            self.backend = "siglip_hf"
            print("[INFO] Real deepfake classifier loaded (SigLIP2, prithivMLmods/deepfake-detector-model-v1)")
        except Exception as e:
            print(f"[WARNING] Could not load real deepfake classifier ({e}) -> using mock Stage-2 detector")
            print("[WARNING] Check: pip install transformers torch, and internet access on first run.")
            self.model = None
            self.processor = None
            self.backend = None

    # ---------- stages ----------
    def _stage1_texture(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = min(1.0, max(0.0, lap_var / 100.0))
        return 1.0 - score  # higher = more "fake-looking" texture

    def _preprocess_efficientnet(self, face_crop):
        face = cv2.resize(face_crop, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face = (face - mean) / std
        return np.transpose(face, (2, 0, 1))

    def _stage2_efficientnet(self, face_crop):
        try:
            import torch
            tensor = torch.from_numpy(self._preprocess_efficientnet(face_crop)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                return float(torch.sigmoid(out).item())
        except Exception as e:
            print(f"[ERROR] EfficientNet inference failed: {e}")
            return 0.5

    def _stage2_siglip(self, face_crop):
        try:
            import torch
            from PIL import Image
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
                deepfake_probability = float(probs[FAKE_LABEL_INDEX].item())
            return deepfake_probability
        except Exception as e:
            print(f"[ERROR] SigLIP deepfake inference failed: {e}")
            return 0.5

    def _stage2_infer(self, face_crop):
        if self.backend == "siglip_hf":
            return self._stage2_siglip(face_crop)
        if self.backend == "efficientnet_local":
            return self._stage2_efficientnet(face_crop)
        # No real model loaded -> mock, clearly labeled as such in the output payload.
        return float(np.random.uniform(0.1, 0.4))

    def _detect_face(self, frame):
        cascade = cv2.CascadeClassifier(self.cascade_path)
        if cascade.empty():
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
        return faces

    # ---------- output ----------
    def _build_output(self, score, stage, latency_ms=0.0, skipped=False):
        window_mean = float(np.mean(self.window)) if self.window else float(score)
        if len(self.window) >= 20:
            recent = list(self.window)[-20:]
            high = sum(1 for s in recent if s > 0.5)
            confidence = "high" if max(high, 20 - high) >= 16 else "low"
        else:
            confidence = "low"
        return {
            "deepfake_probability": window_mean,
            "confidence": confidence,
            "cascade_stage": stage,
            "model_backend": self.backend or "mock",
            "frames_scored_last_30s": len(self.window),
            "latency_ms_avg": round(latency_ms, 1),
            "timestamp": time.time(),
            "skipped": skipped,
        }

    def score_frame(self, frame_bgr):
        start = time.time()
        self.frame_counter += 1
        if self.frame_counter % self.subsample != 0:
            last = self.window[-1] if self.window else 0.5
            return self._build_output(last, 0, skipped=True)

        stage1 = self._stage1_texture(frame_bgr)
        if stage1 < 0.3 or stage1 > 0.7:
            self.window.append(stage1)
            return self._build_output(stage1, 1, (time.time() - start) * 1000)

        faces = self._detect_face(frame_bgr)
        if faces is None or len(faces) == 0:
            self.window.append(stage1)
            return self._build_output(stage1, 1, (time.time() - start) * 1000)

        x, y, w, h = faces[0]
        face_crop = frame_bgr[y:y + h, x:x + w]
        stage2 = self._stage2_infer(face_crop)
        self.window.append(stage2)
        return self._build_output(stage2, 2, (time.time() - start) * 1000)

    def get_visual_deepfake_score(self):
        """0-100 scale for the fusion engine (higher = more likely fake)."""
        if not self.window:
            return 50
        return int(np.mean(self.window) * 100)

    def reset(self):
        self.window.clear()
        self.frame_counter = 0
