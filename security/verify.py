import cv2
import numpy as np
import json
import time
import requests
from datetime import datetime, timezone
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

BACKEND_URL = "http://localhost:5000/score"
SESSION_ID = "live_webrtc_call_xyz_123"

def load_enrolled_embedding(path="enrolled_face.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["embedding"])

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def build_payload(identity_match, similarity_score, face_detected,
                  multiple_faces, confidence, latency_ms):
    identity_score = round(similarity_score * 100)
    verdict = "REAL" if identity_match else "FAKE"
    return {
        "module": "identity",
        "session_id": SESSION_ID,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "face_detected": face_detected,
        "multiple_faces_detected": multiple_faces,
        "identity_score": identity_score,
        "similarity_score": round(similarity_score, 3),
        "threshold_used": 0.6,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "verdict": verdict
    }

def post_to_backend(payload):
    try:
        requests.post(BACKEND_URL, json=payload, timeout=1)
    except Exception:
        pass

def run_verification(enrolled_path="enrolled_face.json", threshold=0.6):
    enrolled_embedding = load_enrolled_embedding(enrolled_path)
    cap = cv2.VideoCapture(0)
    print("=== LIVE VERIFICATION RUNNING ===")
    print("Press ESC to stop.")

    frame_count = 0
    last_payload = None
    last_post_time = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % 15 == 0:
            start = time.time()
            faces = app.get(frame)
            latency_ms = round((time.time() - start) * 1000, 1)

            if len(faces) == 0:
                last_payload = build_payload(False, 0.0, False, False, "none", latency_ms)
            elif len(faces) > 1:
                last_payload = build_payload(False, 0.0, True, True, "none", latency_ms)
            else:
                live_embedding = faces[0].normed_embedding
                score = cosine_similarity(live_embedding, enrolled_embedding)
                score = round(score, 3)

                if score >= threshold:
                    confidence = "high"
                elif score >= 0.50:
                    confidence = "low"
                else:
                    confidence = "none"

                last_payload = build_payload(
                    score >= threshold, score, True, False, confidence, latency_ms
                )

                box = faces[0].bbox.astype(int)
                color = (0, 255, 0) if score >= threshold else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            print(last_payload)

            now = time.time()
            if now - last_post_time >= 2:
                post_to_backend(last_payload)
                last_post_time = now

        if last_payload:
            match = last_payload["verdict"] == "REAL"
            score = last_payload["similarity_score"]
            color = (0, 255, 0) if match else (0, 0, 255)
            cv2.putText(frame, f"Identity: {last_payload['verdict']}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Score: {last_payload['identity_score']}/100", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Confidence: {last_payload['confidence']}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Live Verification", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_verification()