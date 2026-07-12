import cv2
import time
import numpy as np
from collections import deque


def normalize_anomaly_score(
    variance,
    real_baseline_min=6e-5,
    virtual_typical=4.5e-7
):
    if variance >= real_baseline_min:
        return 0.0

    if variance <= virtual_typical:
        return 1.0

    score = (
        real_baseline_min - variance
    ) / (
        real_baseline_min - virtual_typical
    )

    return round(score, 2)


def get_timing_anomaly_score(
    camera_index=0,
    duration_sec=3,
    window_size=50
):

    cap = cv2.VideoCapture(camera_index)

    # camera failed
    if not cap.isOpened():
        cap.release()
        return None


    deltas = deque(maxlen=window_size)

    prev_t = time.time()
    start = time.time()
    last_variance = None


    while time.time() - start < duration_sec:

        ok, frame = cap.read()

        if not ok:
            continue


        now = time.time()

        dt = now - prev_t
        prev_t = now

        deltas.append(dt)


        if len(deltas) >= 10:
            last_variance = float(np.var(deltas))


    cap.release()


    # no usable frames received
    if last_variance is None:
        return None


    return normalize_anomaly_score(last_variance)
