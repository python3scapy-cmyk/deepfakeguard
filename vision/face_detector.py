"""
Face detection + multi-signal anti-spoof analysis.
Uses OpenCV Haar Cascade (auto-downloaded) for detection, and a
weighted combination of texture / moire / reflection / depth /
noise / color-banding / screen-bezel / edge-artifact checks to
flag printed photos, phone/tablet screens, and similar 2D spoofs.
"""
import os
import urllib.request

import cv2
import numpy as np


def _has_faces(faces):
    """Safe truthiness check for cv2 face results (avoids numpy ambiguity)."""
    if faces is None:
        return False
    if isinstance(faces, np.ndarray):
        return faces.size > 0
    return len(faces) > 0


class AntiSpoofDetector:
    """Runs several independent 2D-spoof heuristics and combines them."""

    def analyze(self, frame, face_roi, faces):
        if face_roi is None or face_roi.size == 0:
            return {
                "is_spoof": True,
                "confidence": 1.0,
                "score": 0.0,
                "reasons": ["no_face"],
                "details": {},
            }

        results = {
            "laplacian": self._check_laplacian(face_roi),
            "moire": self._check_moire(frame, faces),
            "reflection": self._check_reflection(frame, faces),
            "depth": self._check_depth_cue(frame, faces),
            "noise": self._check_noise_pattern(face_roi),
            "color_banding": self._check_color_banding(face_roi),
            "screen_bezel": self._check_screen_bezel(frame, faces),
            "edge_artifacts": self._check_edge_artifacts(frame, faces),
        }

        weights = {
            "laplacian": 0.10,
            "moire": 0.15,
            "reflection": 0.15,
            "depth": 0.10,
            "noise": 0.10,
            "color_banding": 0.10,
            "screen_bezel": 0.20,
            "edge_artifacts": 0.10,
        }

        total_score = 0.0
        total_weight = 0.0
        reasons = []
        for name, data in results.items():
            w = weights.get(name, 0)
            total_score += data["score"] * w
            total_weight += w
            if data.get("is_spoof") and data.get("reason"):
                reasons.append(data["reason"])

        final_score = total_score / total_weight if total_weight > 0 else 0.5

        # Strong indicators (screen bezel, moire, glass reflection) are each
        # individually noisy on a real webcam under imperfect lighting.
        # Only override the weighted average when at least TWO of them agree
        # -- a single flaky heuristic shouldn't be able to condemn a real
        # user by itself.
        strong = ("screen_bezel", "moire_pattern", "glass_reflection")
        strong_hits = sum(1 for r in reasons if any(s in r for s in strong))
        if strong_hits >= 2:
            final_score = min(final_score, 0.3)

        is_spoof = final_score < 0.45
        confidence = 1.0 - abs(final_score - 0.5) * 2

        return {
            "is_spoof": bool(is_spoof),
            "confidence": float(confidence),
            "score": float(final_score),
            "reasons": reasons if reasons else ["natural"],
            "details": results,
        }

    def _check_laplacian(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        score = min(1.0, lap_var / 20.0)
        is_spoof = lap_var < 5.0
        return {"score": float(score), "is_spoof": bool(is_spoof),
                "reason": "flat_texture" if is_spoof else None, "value": float(lap_var)}

    def _check_moire(self, frame, faces):
        if not _has_faces(faces):
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        x, y, w, h = faces[0]
        x1, y1 = max(0, x - 40), max(0, y - 40)
        x2 = min(frame.shape[1], x + w + 40)
        y2 = min(frame.shape[0], y + h + 40)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray.astype(np.float32))
        magnitude = np.abs(np.fft.fftshift(f))
        magnitude = magnitude / (magnitude.max() + 1e-10)
        hh, ww = magnitude.shape
        cy, cx = hh // 2, ww // 2
        Y, X = np.ogrid[:hh, :ww]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = (dist > 2) & (dist < min(hh, ww) // 3)
        if np.sum(mask) == 0:
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        peak_ratio = np.sum(magnitude[mask] > 0.20) / np.sum(mask)
        is_spoof = peak_ratio > 0.003
        score = 1.0 - min(1.0, peak_ratio * 100)
        return {"score": float(score), "is_spoof": bool(is_spoof),
                "reason": "moire_pattern" if is_spoof else None, "value": float(peak_ratio)}

    def _check_reflection(self, frame, faces):
        if not _has_faces(faces):
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        x, y, w, h = faces[0]
        x1, y1 = max(0, x - 15), max(0, y - 15)
        x2 = min(frame.shape[1], x + w + 15)
        y2 = min(frame.shape[0], y + h + 15)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        thresh = np.percentile(v, 90)
        bright_mask = v > thresh * 0.95
        bright_ratio = np.sum(bright_mask) / bright_mask.size
        # Real faces under normal indoor/webcam lighting always have some
        # specular highlight on the forehead/nose -- that's not a "glass
        # reflection". Only flag genuinely large, sharp bright regions
        # (the kind a phone/tablet screen or glass surface produces).
        is_spoof = bright_ratio > 0.06
        score = 1.0 - min(1.0, bright_ratio * 12)
        return {"score": float(score), "is_spoof": bool(is_spoof),
                "reason": "glass_reflection" if is_spoof else None, "value": float(bright_ratio)}

    def _check_depth_cue(self, frame, faces):
        if not _has_faces(faces):
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        x, y, w, h = faces[0]
        face_roi = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        ny, nx = int(h * 0.35), int(w * 0.35)
        nw, nh = int(w * 0.3), int(h * 0.25)
        nose_roi = gray[ny:ny + nh, nx:nx + nw]
        cy, ch = int(h * 0.4), int(h * 0.25)
        left_cheek = gray[cy:cy + ch, 0:int(w * 0.2)]
        right_cheek = gray[cy:cy + ch, int(w * 0.8):w]
        nose_lap = cv2.Laplacian(nose_roi, cv2.CV_64F).var() if nose_roi.size > 0 else 0
        left_lap = cv2.Laplacian(left_cheek, cv2.CV_64F).var() if left_cheek.size > 0 else 1
        right_lap = cv2.Laplacian(right_cheek, cv2.CV_64F).var() if right_cheek.size > 0 else 1
        avg_cheek = (left_lap + right_lap) / 2.0
        depth_ratio = nose_lap / (avg_cheek + 1e-10)
        # Real webcam faces routinely have uneven lighting (one cheek in
        # shadow, etc.) which shifts this ratio around a lot -- only flag
        # genuinely extreme flatness, not ordinary lighting variation.
        is_spoof = depth_ratio < 0.5 and nose_lap > 3
        score = min(1.0, depth_ratio / 1.3) if depth_ratio > 0 else 0.5
        return {"score": float(score), "is_spoof": bool(is_spoof),
                "reason": "flat_depth" if is_spoof else None, "value": float(depth_ratio)}

    def _check_noise_pattern(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0)
        noise = gray.astype(np.float32) - blurred
        noise_std = np.std(noise.flatten())
        is_spoof = noise_std < 1.5
        score = min(1.0, noise_std / 5.0)
        return {"score": float(score), "is_spoof": bool(is_spoof),
                "reason": "artificial_noise" if is_spoof else None, "value": float(noise_std)}

    def _check_color_banding(self, face_roi):
        if len(face_roi.shape) != 3:
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        ratios = []
        for i in range(3):
            channel = face_roi[:, :, i].astype(np.float32)
            ratios.append(len(np.unique(channel.astype(np.uint8))) / channel.size)
        avg_unique = float(np.mean(ratios))
        is_spoof = avg_unique < 0.10
        score = min(1.0, avg_unique / 0.20)
        return {"score": float(score), "is_spoof": bool(is_spoof),
                "reason": "color_banding" if is_spoof else None, "value": avg_unique}

    def _check_screen_bezel(self, frame, faces):
        if not _has_faces(faces):
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        x, y, w, h = faces[0]
        fh, fw = frame.shape[:2]
        margin = 25
        areas = []
        if min(margin, y) > 0:
            areas.append(frame[max(0, y - margin):y, x:x + w])
        if min(margin, fh - (y + h)) > 0:
            areas.append(frame[y + h:y + h + min(margin, fh - (y + h)), x:x + w])
        if min(margin, x) > 0:
            areas.append(frame[y:y + h, max(0, x - margin):x])
        if min(margin, fw - (x + w)) > 0:
            areas.append(frame[y:y + h, x + w:x + w + min(margin, fw - (x + w))])
        areas = [a for a in areas if a is not None and a.size > 0]
        if not areas:
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        dark_ratios = []
        for area in areas:
            gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
            dark_ratios.append(np.sum(gray < 35) / gray.size)
        avg_dark = float(np.mean(dark_ratios))
        is_spoof = avg_dark > 0.40
        score = 1.0 - min(1.0, avg_dark * 1.3)
        return {"score": float(score), "is_spoof": bool(is_spoof),
                "reason": "screen_bezel" if is_spoof else None, "value": avg_dark}

    def _check_edge_artifacts(self, frame, faces):
        if not _has_faces(faces):
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        x, y, w, h = faces[0]
        fh, fw = frame.shape[:2]
        samples = []
        if y > 0:
            samples.append(frame[y, max(0, x - 5):min(fw, x + w + 5)])
        if y + h < fh:
            samples.append(frame[y + h - 1, max(0, x - 5):min(fw, x + w + 5)])
        edge_scores = []
        for edge in samples:
            if edge.size < 3:
                continue
            gray = cv2.cvtColor(edge.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).flatten()
            grad = np.abs(np.diff(gray.astype(np.float32)))
            edge_scores.append(np.mean(grad))
        if not edge_scores:
            return {"score": 0.5, "is_spoof": False, "reason": None, "value": 0}
        avg_edge = float(np.mean(edge_scores))
        is_spoof = avg_edge > 25.0
        score = 1.0 - min(1.0, avg_edge / 40.0)
        return {"score": float(score), "is_spoof": bool(is_spoof),
                "reason": "sharp_edge" if is_spoof else None, "value": avg_edge}


class FaceLandmarkDetector:
    """Haar-cascade face detection with frame-skipping for speed,
    plus attached anti-spoof analysis on the primary face."""

    def __init__(self):
        self.xml_path = self._resolve_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(self.xml_path)
        if self.face_cascade.empty():
            print("[ERROR] Cascade classifier failed to load!")
        self.frame_skip = 1
        self.frame_count = 0
        self.last_faces = []
        self.last_payload = {
            "face_count": 0, "landmarks": [], "status": "NO_FACE",
            "texture_score": 0.0, "spoof_analysis": None,
        }
        self.anti_spoof = AntiSpoofDetector()

    def _resolve_cascade_path(self):
        """
        Prefer the Haar cascade XML that ships inside the installed
        opencv-python(-headless) package (cv2.data.haarcascades) -- no
        internet needed and it always matches the installed cv2 build.
        Falls back to a local copy or a GitHub download only if that
        bundled path is somehow missing.
        """
        local_name = "haarcascade_frontalface_default.xml"

        bundled = os.path.join(getattr(cv2.data, "haarcascades", ""), local_name)
        if bundled and os.path.exists(bundled):
            return bundled

        if os.path.exists(local_name):
            return local_name

        try:
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/"
                "data/haarcascades/haarcascade_frontalface_default.xml",
                local_name,
            )
            print(f"[INFO] Downloaded {local_name}")
            return local_name
        except Exception as e:
            print(f"[WARNING] Could not find or download cascade XML: {e}")
            return local_name

    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return self.last_faces, self.last_payload

        payload = {
            "face_count": 0, "landmarks": [], "status": "NO_FACE",
            "texture_score": 0.0, "spoof_analysis": None,
        }
        if frame is None or frame.size == 0:
            return self.last_faces, payload

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Histogram equalization makes the cascade far more robust to
        # webcam exposure/lighting variation -- this is the single biggest
        # lever for "real human standing in front of a normal-lit camera
        # doesn't get detected" issues with Haar cascades.
        gray = cv2.equalizeHist(gray)
        scale = 0.5
        small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

        faces = self.face_cascade.detectMultiScale(
            small_gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(30, 30), maxSize=(int(w * scale), int(h * scale)),
        )
        faces = [(int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
                 for (x, y, fw, fh) in faces]

        face_count = len(faces)
        payload["face_count"] = face_count

        if face_count == 0:
            payload["status"] = "NO_FACE"
        elif face_count > 1:
            payload["status"] = "MULTIPLE_FACES_DETECTED"
        else:
            payload["status"] = "VALID_SESSION"
            x, y, bw, bh = faces[0]
            payload["landmarks"] = [x, y, bw, bh]
            face_roi = frame[y:y + bh, x:x + bw]
            analysis = self.anti_spoof.analyze(frame, face_roi, faces)
            payload["spoof_analysis"] = analysis
            payload["texture_score"] = analysis["details"].get("laplacian", {}).get("value", 0.0)

        self.last_faces = faces
        self.last_payload = payload
        return faces, payload

    def draw_landmarks(self, frame, faces, payload):
        status = payload.get("status", "NO_FACE")
        analysis = payload.get("spoof_analysis")

        if status == "MULTIPLE_FACES_DETECTED":
            cv2.putText(frame, "ALERT: MULTIPLE FACES", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        elif status == "NO_FACE":
            cv2.putText(frame, "ALERT: NO FACE", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif status == "VALID_SESSION":
            for (x, y, w, h) in faces:
                if analysis and analysis.get("is_spoof"):
                    color = (0, 0, 255)
                    label = f"SPOOF ({analysis.get('confidence', 0):.2f})"
                    reasons = [r for r in analysis.get("reasons", []) if r]
                    if reasons:
                        cv2.putText(frame, f"Why: {'|'.join(reasons)[:30]}", (x, y - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                else:
                    color = (0, 255, 0)
                    label = "REAL HUMAN"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def close(self):
        pass
