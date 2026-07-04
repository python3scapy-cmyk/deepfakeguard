import cv2
import numpy as np
import json
import time
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

def load_enrolled_embedding(path="enrolled_face.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["embedding"])

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def run_verification(enrolled_path="enrolled_face.json", threshold=0.6):
    enrolled_embedding = load_enrolled_embedding(enrolled_path)
    cap = cv2.VideoCapture(0)
    print("=== LIVE VERIFICATION RUNNING ===")
    print("Press ESC to stop.")

    frame_count = 0
    last_payload = None

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
                last_payload = {
                    "identity_match": False,
                    "similarity_score": 0.0,
                    "threshold_used": threshold,
                    "face_detected": False,
                    "multiple_faces_detected": False,
                    "confidence": "none",
                    "latency_ms": latency_ms,
                    "timestamp": time.time()
                }
            elif len(faces) > 1:
                last_payload = {
                    "identity_match": False,
                    "similarity_score": 0.0,
                    "threshold_used": threshold,
                    "face_detected": True,
                    "multiple_faces_detected": True,
                    "confidence": "none",
                    "latency_ms": latency_ms,
                    "timestamp": time.time()
                }
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

                last_payload = {
                    "identity_match": score >= threshold,
                    "similarity_score": score,
                    "threshold_used": threshold,
                    "face_detected": True,
                    "multiple_faces_detected": False,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "timestamp": time.time()
                }

                box = faces[0].bbox.astype(int)
                color = (0, 255, 0) if score >= threshold else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

        if last_payload:
            match = last_payload["identity_match"]
            score = last_payload["similarity_score"]
            status = "MATCH" if match else "NO MATCH"
            color = (0, 255, 0) if match else (0, 0, 255)
            cv2.putText(frame, f"Identity: {status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Score: {score}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Confidence: {last_payload['confidence']}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            print(last_payload)

        cv2.imshow("Live Verification", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_verification()