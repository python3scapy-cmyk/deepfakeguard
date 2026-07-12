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
import statistics
import time
from collections import deque

import cv2
import numpy as np

DEFAULT_MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
# Optional second opinion for ensemble mode (ViT fine-tuned on a different
# deepfake corpus). Averaging two ARCHITECTURES is the strongest cheap fix
# for single-model false positives on webcam JPEG frames.
ENSEMBLE_MODEL_NAME = "dima806/deepfake_vs_real_image_detection"
# This model's label space: 0 = fake, 1 = real (confirmed on the model card).
FAKE_LABEL_INDEX = 0
REAL_LABEL_INDEX = 1


class DeepfakeDetector:
    def __init__(self, model_path=None, device="cpu",
                 hf_model_name=DEFAULT_MODEL_NAME, flip_labels=False,
                 ensemble=False):
        self.device = device
        # If real AI/fake content is consistently coming out as HUMAN (or
        # vice versa), the model card's documented label order (0=fake,
        # 1=real) may not match this checkpoint's actual output for your
        # test images. Flip this (main.py: --flip-deepfake-labels) to swap
        # which softmax index counts as "fake" -- watch the [DEEPFAKE]
        # debug line below to see the raw probabilities and confirm.
        self.flip_labels = flip_labels
        self._last_debug_print = 0.0
        self._last_noface_warn = 0.0
        self.frame_counter = 0
        # NOTE: this used to default to 3 (only score every 3rd call). Now
        # that score_frame() runs inside main.py's AsyncWorker on its own
        # background thread, the worker ALREADY only ever processes the
        # most recent submitted frame and naturally paces itself to real
        # inference speed (dropping frames it can't keep up with) -- so
        # this extra gate was pure redundant delay on top of that, making
        # stage2_window take 3x longer to fill with real evidence than
        # necessary. 1 = score every frame the worker actually gets to.
        self.subsample = 1
        self.window = deque(maxlen=30)
        # Separate from `window`: only ever holds scores that came from a
        # REAL model inference (stage 2, backend != None). `window` mixes
        # in cheap stage-1 texture values and mock placeholders, which is
        # fine for the rolling overlay number but is exactly the wrong
        # thing to threshold a HUMAN/AI verdict on -- a texture heuristic
        # or `np.random.uniform(0.1, 0.4)` mock score should never be able
        # to produce an "AI DETECTED" banner. main.py's classify_human_ai
        # reads this window instead of `window`.
        self.stage2_window = deque(maxlen=8)
        self.model = None
        self.processor = None
        self.models = []  # [(name, model, processor, fake_label_idx), ...]
        # "siglip_hf", "ensemble_xN", "efficientnet_local", or None (mock)
        self.backend = None
        bundled = os.path.join(
    getattr(
        cv2.data,
        "haarcascades",
        ""),
         "haarcascade_frontalface_default.xml")
        self.cascade_path = bundled if os.path.exists(
            bundled) else "haarcascade_frontalface_default.xml"

        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

        if self.face_cascade.empty():
            print(
    f"[WARNING] Could not load Haar cascade from {
        self.cascade_path}")
        # Priority 1: a local custom checkpoint explicitly passed in (e.g. if
        # someone later trains/exports their own EfficientNet weights).
        if model_path and os.path.exists(model_path):
            try:
                import torch
                from efficientnet_pytorch import EfficientNet
                self.model = EfficientNet.from_name(
                    "efficientnet-b0", num_classes=1)
                self.model.load_state_dict(
                    torch.load(model_path, map_location=device))
                self.model.to(device)
                self.model.eval()
                self.backend = "efficientnet_local"
                print(
    f"[INFO] Loaded local EfficientNet checkpoint from {model_path}")
                return
            except Exception as e:
                print(
    f"[WARNING] Could not load local EfficientNet checkpoint: {e}")
                self.model = None

        # Priority 2: real pretrained deepfake classifier(s) from Hugging
        # Face (auto-download on first run, cached afterwards). Each
        # checkpoint's fake-label index is read from its OWN config
        # (id2label), so adding/swapping checkpoints can't silently invert
        # REAL/FAKE; --flip-deepfake-labels still overrides per run.
        try:
            import torch  # noqa: F401 - verified importable before loading
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            checkpoints = [hf_model_name]
            if ensemble:
                checkpoints.append(ENSEMBLE_MODEL_NAME)

            for ck in checkpoints:
                try:
                    print(f"[INFO] Loading deepfake classifier '{ck}' "
                          f"(first run downloads weights)...")
                    proc = AutoImageProcessor.from_pretrained(ck)
                    mdl = AutoModelForImageClassification.from_pretrained(ck)
                    mdl.to(device)
                    mdl.eval()
                    id2label = getattr(mdl.config, "id2label", {}) or {}
                    fake_idx = None
                    for i, lbl in id2label.items():
                        if "fake" in str(lbl).lower():
                            fake_idx = int(i)
                            break
                    if fake_idx is None:
                        fake_idx = FAKE_LABEL_INDEX
                    if self.flip_labels:
                        fake_idx = 1 - fake_idx
                    self.models.append((ck, mdl, proc, fake_idx))
                    print(f"[INFO] Loaded '{ck}' (fake label index {fake_idx})")
                except Exception as e:
                    print(f"[WARNING] Could not load checkpoint '{ck}': {e}")

            if self.models:
                self.model, self.processor = self.models[0][1], self.models[0][2]
                self.backend = ("siglip_hf" if len(self.models) == 1
                                else "ensemble_x{}".format(len(self.models)))
                print(f"[INFO] Deepfake backend ready: {self.backend}")
            else:
                raise RuntimeError("no HF checkpoint could be loaded")
        except Exception as e:
            print(
    f"[WARNING] Could not load real deepfake classifier ({e}) -> using mock Stage-2 detector")
            print(
                "[WARNING] Check: pip install transformers torch, and internet access on first run.")
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
            tensor = torch.from_numpy(
    self._preprocess_efficientnet(face_crop)).float().unsqueeze(0).to(
        self.device)
            with torch.no_grad():
                out = self.model(tensor)
                return float(torch.sigmoid(out).item())
        except Exception as e:
            print(f"[ERROR] EfficientNet inference failed: {e}")
            return 0.5

    def _stage2_hf(self, face_crop):
        """Score with EVERY loaded HF checkpoint, on the original AND the
        horizontally-flipped crop (test-time augmentation), and average.
        TTA + cross-architecture averaging is the cheap, reliable way to
        smooth out the single-frame false positives that webcam JPEG noise
        and lighting produce with any single model."""
        try:
            import torch
            from PIL import Image
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            variants = [Image.fromarray(rgb),
                        Image.fromarray(rgb[:, ::-1].copy())]
            per_model = []
            with torch.no_grad():
                for name, mdl, proc, fake_idx in self.models:
                    vals = []
                    for img in variants:
                        inputs = proc(images=img, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        out = mdl(**inputs)
                        pr = torch.nn.functional.softmax(out.logits, dim=1).squeeze()
                        vals.append(float(pr[fake_idx].item()))
                    per_model.append((name, float(np.mean(vals))))
            prob = float(np.mean([v for _n, v in per_model]))

            # Throttled debug line: raw per-model probabilities, so a
            # mislabeled/underperforming checkpoint is visible immediately
            # instead of silently skewing the HUMAN/AI verdict.
            now = time.time()
            if now - self._last_debug_print > 1.0:
                detail = "  ".join(f"{n.split('/')[-1]}={v:.3f}" for n, v in per_model)
                print(f"[DEEPFAKE] p(fake)={prob:.3f}  [{detail}] "
                      f"flip_labels={self.flip_labels}")
                self._last_debug_print = now
            return prob
        except Exception as e:
            print(f"[ERROR] HF deepfake inference failed: {e}")
            return 0.5

    def _stage2_infer(self, face_crop):
        if self.models:
            return self._stage2_hf(face_crop)
        if self.backend == "efficientnet_local":
            return self._stage2_efficientnet(face_crop)
        # No real model loaded -> mock, clearly labeled as such in the output
        # payload.
        return float(np.random.uniform(0.1, 0.4))

    def _crop_with_margin(self, frame, box, margin=0.25):
        """Expand the face box by `margin` on every side before cropping.
        These classifiers were fine-tuned on loosely-framed face images; a
        tight eyebrows-to-chin crop measurably shifts them toward random."""
        x, y, w, h = box
        H, W = frame.shape[:2]
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(W, x + w + mx), min(H, y + h + my)
        return frame[y0:y1, x0:x1]

    def analyze_single_frame(self, frame_bgr):
        """One-shot scoring for offline (uploaded photo/video) analysis --
        always runs Stage 2 directly on the detected face, bypassing the
        live-session cascade's texture-gate/frame-subsampling/rolling-window
        logic (those exist for real-time streams, not a one-off upload)."""
        faces = self._detect_face(frame_bgr)
        if faces is None or len(faces) == 0:
            return {"face_found": False, "deepfake_probability": None}
        face_crop = self._crop_with_margin(frame_bgr, faces[0])
        prob = self._stage2_infer(face_crop)
        return {"face_found": True, "deepfake_probability": float(prob)}

    def _detect_face(self, frame):
        if self.face_cascade.empty():
            return None
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    
        faces = self.face_cascade.detectMultiScale(
            gray,
            1.1,
            4,
            minSize=(80, 80)
        )
    
        return faces

    # ---------- output ----------
    def _build_output(self, score, stage, latency_ms=0.0, skipped=False):
        # Median rather than mean: one bad frame (motion blur, lighting
        # flicker, a hand passing in front of the camera) can push a mean
        # far off, and that single spike was exactly what caused sporadic
        # false "AI DETECTED" flashes. A median is unmoved by one outlier
        # in a 30-sample window.
        window_mean = float(np.median(self.window)) if self.window else float(score)
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

    def score_frame(self, frame_bgr, face_box=None):
        """face_box, if given, is an (x, y, w, h) tuple from a face
        detector that already ran upstream (main.py passes the same box
        FaceLandmarkDetector found for this frame) -- reusing it avoids a
        second, independent Haar cascade call and keeps every module
        looking at the same face region. Falls back to this class's own
        cascade if not given (e.g. analyze_single_frame's offline path)."""
        start = time.time()
        self.frame_counter += 1
        if self.frame_counter % self.subsample != 0:
            last = self.window[-1] if self.window else 0.5
            return self._build_output(last, 0, skipped=True)

        stage1 = self._stage1_texture(frame_bgr)

        # With a real model loaded, ALWAYS run it -- do not gate on the
        # Stage-1 texture heuristic. That heuristic was meant as a cheap
        # pre-filter to skip the (once-blocking) real model on "obviously
        # fine" frames, but in practice most ordinary camera frames' Laplacian
        # variance falls OUTSIDE the [0.3, 0.7] "ambiguous" band (crisp frame
        # -> near 1, low-detail/blurry -> near 0) -- so the real classifier
        # almost never ran, and the system was quietly falling back to the
        # texture score for nearly every frame. That's why AI content wasn't
        # being flagged: the real detector was being skipped, not making a
        # wrong call. Now that inference runs on a background thread (see
        # main.py's AsyncWorker), there's no performance reason left to gate
        # it -- the texture-only shortcut is kept ONLY for when no real
        # model could be loaded at all (mock backend).
        has_real_model = self.backend in ("siglip_hf", "efficientnet_local")
        if not has_real_model and (stage1 < 0.3 or stage1 > 0.7):
            self.window.append(stage1)
            return self._build_output(stage1, 1, (time.time() - start) * 1000)

        if face_box is not None:
            faces = [face_box]
        else:
            faces = self._detect_face(frame_bgr)
        if faces is None or len(faces) == 0:
            if has_real_model:
                now = time.time()
                if now - self._last_noface_warn > 2.0:
                    print("[DEEPFAKE] no face found this cycle -- Stage-2 model was "
                          "NOT run (nothing to score). If you're holding up a phone/photo "
                          "and this keeps printing, screen glare/angle is likely defeating "
                          "face detection -- try flatter angle, less glare, or move closer.")
                    self._last_noface_warn = now
            self.window.append(stage1)
            return self._build_output(stage1, 1, (time.time() - start) * 1000)

        face_crop = self._crop_with_margin(frame_bgr, faces[0])
        if face_crop.size == 0:
            self.window.append(stage1)
            return self._build_output(stage1, 1, (time.time() - start) * 1000)

        stage2 = self._stage2_infer(face_crop)
        self.window.append(stage2)
        if has_real_model:
            # Only real inference results count as "evidence" for the
            # HUMAN/AI verdict -- never a mock score (backend is None
            # in that case, so this branch simply doesn't run).
            self.stage2_window.append(stage2)
        return self._build_output(stage2, 2, (time.time() - start) * 1000)

    def get_stage2_window(self):
        """Real (non-mock) Stage-2 model scores from the last ~20 scored
        frames, oldest first. Empty until the model has actually run --
        callers must not treat an empty list as 'human', only as 'no
        evidence yet'."""
        return list(self.stage2_window)

    def get_visual_deepfake_score(self):
        """0-100 scale for the fusion engine (higher = more likely fake).
        Median of the rolling window, same outlier-resistance rationale as
        _build_output."""
        if not self.window:
            return 50
        return int(statistics.median(self.window) * 100)

    def reset(self):
        self.window.clear()
        self.stage2_window.clear()
        self.frame_counter = 0
