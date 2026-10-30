"""Privacy filter: blur detected face regions on the video stream.

Off by default. When active, only event metadata is persisted.
"""
from __future__ import annotations

import cv2
import numpy as np

_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        try:
            _CASCADE = cv2.CascadeClassifier(path)
        except Exception:
            _CASCADE = False
    return _CASCADE


def blur_faces(frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
    cas = _cascade()
    if not cas:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cas.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
    out = frame.copy()
    ksize = (31, 31)
    for (x, y, w, h) in faces:
        roi = out[y : y + h, x : x + w]
        roi = cv2.GaussianBlur(roi, ksize, 0)
        out[y : y + h, x : x + w] = roi
    return out