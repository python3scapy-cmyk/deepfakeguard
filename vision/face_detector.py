import logging
import time
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

class FaceLandmarkDetector:
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def process_frame(self, frame: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        payload = {"face_count": 0, "landmarks": [], "fps": 0.0, "timestamp": time.time(), "status": "NO_FACE", "texture_score": 0.0}
        if frame is None or frame.size == 0: return None, payload
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        face_count = len(faces)
        payload["face_count"] = face_count
        if face_count == 0: payload["status"] = "NO_FACE"
        elif face_count > 1: payload["status"] = "MULTIPLE_FACES_DETECTED"
        else:
            payload["status"] = "VALID_SESSION"
            x, y, w, h = faces[0]
            payload["landmarks"] = [int(x), int(y), int(w), int(h)]
            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size > 0:
                laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
                payload["texture_score"] = float(laplacian_var)
        return faces, payload

    def draw_landmarks(self, frame: np.ndarray, faces, payload: Dict[str, Any], texture_score: float = 0.0) -> np.ndarray:
        status = payload["status"]
        current_score = payload["texture_score"] if texture_score == 0.0 else texture_score
        if status == "MULTIPLE_FACES_DETECTED":
            cv2.putText(frame, "SECURITY ALERT: MULTIPLE FACES DETECTED", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for (x, y, w, h) in faces: cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        elif status == "NO_FACE":
            cv2.putText(frame, "SECURITY ALERT: NO FACE DETECTED", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif status == "VALID_SESSION":
            for (x, y, w, h) in faces:
                is_real = current_score > 70.0
                color = (0, 255, 0) if is_real else (0, 0, 255)
                box_label = "REAL HUMAN" if is_real else "SUSPECTED SPOOF (PHOTO)"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{box_label} ^(Tex: {current_score:.1f}^)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    def close(self) -> None: pass
