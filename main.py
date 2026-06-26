import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from security.security_module import get_security_signal, compute_security_score
    SECURITY_AVAILABLE = True
except ImportError as e:
    SECURITY_AVAILABLE = False
    print(f"[WARNING] Security module not found: {e}")

from vision.face_detector import FaceLandmarkDetector
from audio_module.train_xgb import start_training


def main():
    print("=== MULTIMODAL ANTI-SPOOFING SYSTEM ===")
    print("[1] Fast Test Mode (Skip audio training, opens camera in 2 seconds)")
    print("[2] Production Mode (Train XGBoost on 1.14 GB dataset)")
    choice = input("Select Mode (1 or 2): ").strip()

    if choice == "1":
        audio_real_probability = 0.95
    else:
        audio_model = start_training()
        mock_audio_features = np.random.rand(1, 26)
        audio_real_probability = float(audio_model.predict_proba(mock_audio_features)[0][1])


    if SECURITY_AVAILABLE:
        try:
            sec = get_security_signal(camera_index=0)
            security_score = compute_security_score(sec)
            print(f"\n[SECURITY] Camera name: {sec['device_name']}")
            print(f"[SECURITY] Is fake? {sec['virtual_camera_detected']}")
            print(f"[SECURITY] Score: {security_score}\n")
        except Exception as e:
            print(f"[WARNING] Security check failed: {e}")
            security_score = 1.0
            sec = {"device_name": "UNKNOWN", "virtual_camera_detected": False}
    else:
        security_score = 1.0
        sec = {"device_name": "UNKNOWN", "virtual_camera_detected": False}

    detector = FaceLandmarkDetector()
    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 30.0
        prev_time = current_time

        faces, vision_payload = detector.process_frame(frame)
        vision_payload["fps"] = int(fps)
        t_score = vision_payload["texture_score"]

        # ============================================
        # MEMBER A: New trust score with security
        # ============================================
        if vision_payload["status"] == "VALID_SESSION" and t_score > 70.0:
            vision_score = 0.92
            final_trust_score = (
                0.50 * audio_real_probability +
                0.30 * vision_score +
                0.20 * security_score
            )
        else:
            final_trust_score = 0.0

        # ============================================
        # MEMBER A: Decision logic with virtual camera block
        # ============================================
        if sec.get("virtual_camera_detected", False):
            system_status = "ACCESS DENIED - VIRTUAL CAMERA"
            status_color = (0, 0, 255)

        elif vision_payload["status"] == "VALID_SESSION" and final_trust_score >= 0.50:
            system_status = "ACCESS GRANTED"
            status_color = (0, 255, 0)

        else:
            if vision_payload["status"] == "MULTIPLE_FACES_DETECTED":
                system_status = "ACCESS DENIED - MULTIPLE USERS"
            elif vision_payload["status"] == "NO_FACE":
                system_status = "ACCESS DENIED - NO USER"
            else:
                system_status = "ACCESS DENIED - SPOOF DETECTED"
            status_color = (0, 0, 255)

        frame = detector.draw_landmarks(frame, faces, vision_payload, t_score)

        cv2.putText(frame, f"FPS: {vision_payload['fps']}", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.putText(frame, f"SYSTEM: {system_status}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        cv2.putText(frame, f"DEVICE: {sec['device_name']}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, f"SEC SCORE: {security_score:.2f}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(frame, f"TRUST: {final_trust_score:.2f}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Multimodal Anti-Spoofing Fusion Engine", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
