import cv2
import numpy as np
import json
import os
from insightface.app import FaceAnalysis

DEFAULT_ENROLLED_PATH = os.path.join(
    os.path.dirname(__file__),
    "enrolled_face.json"
)

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def enroll_face(save_path="enrolled_face.json"):
    cap = cv2.VideoCapture(0)
    print("=== FACE ENROLLMENT ===")
    print("Press SPACE to capture your face, ESC to cancel.")

    enrolled = False
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        faces = app.get(frame)

        if len(faces) == 1:
            face = faces[0]
            box = face.bbox.astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, "Face detected - press SPACE to enroll",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(faces) == 0:
            cv2.putText(frame, "No face detected",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Multiple faces - only one allowed",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Enrollment", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Enrollment cancelled.")
            break

        if key == 32 and len(faces) == 1:
            face = faces[0]
            box = face.bbox.astype(int)
            w = box[2] - box[0]
            h = box[3] - box[1]

            if w < 80 or h < 80:
                print("Face too small - move closer to the camera.")
                continue

            embedding = face.normed_embedding.tolist()
            with open(save_path, "w") as f:
                json.dump({"embedding": embedding}, f)

            print(f"Enrollment successful! Embedding saved to {save_path}")
            enrolled = True
            break

    cap.release()
    cv2.destroyAllWindows()
    return enrolled

if __name__ == "__main__":
    enroll_face()
