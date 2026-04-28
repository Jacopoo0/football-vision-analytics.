import cv2
import json
import numpy as np
from sklearn.cluster import KMeans

JERSEY_Y_TOP    = 0.15
JERSEY_Y_BOTTOM = 0.55
JERSEY_X_LEFT   = 0.15
JERSEY_X_RIGHT  = 0.85
UNKNOWN_MARGIN  = 0.25


def _dominant_lab(frame_bgr: np.ndarray, bbox: tuple):
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if h < 20 or w < 10:
        return None

    jy1 = int(y1 + h * JERSEY_Y_TOP)
    jy2 = int(y1 + h * JERSEY_Y_BOTTOM)
    jx1 = int(x1 + w * JERSEY_X_LEFT)
    jx2 = int(x1 + w * JERSEY_X_RIGHT)

    crop = frame_bgr[jy1:jy2, jx1:jx2]
    if crop.size == 0:
        return None

    hsv        = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([90, 255, 255]))
    valid      = cv2.bitwise_not(grass_mask)
    pixels     = crop[valid > 0]

    if len(pixels) < 15:
        pixels = crop.reshape(-1, 3)

    pixels_u8  = pixels.reshape(-1, 1, 3).astype(np.uint8)
    pixels_lab = cv2.cvtColor(pixels_u8, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    k  = min(3, len(pixels_lab))
    if k < 2:
        return pixels_lab.mean(axis=0)

    km = KMeans(n_clusters=k, n_init=3, random_state=0)
    km.fit(pixels_lab)
    best = np.argmax(np.bincount(km.labels_))
    return km.cluster_centers_[best]


class TeamClassifier:
    def __init__(self):
        self.centroids      = {}
        self.unknown_margin = UNKNOWN_MARGIN

    def load_samples(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)
        for key, samples in data.items():
            tid = int(str(key).split("_")[-1])
            arr = np.array(samples, dtype=np.float32)
            self.centroids[tid] = arr.mean(axis=0)
        print(f"TeamClassifier: {len(self.centroids)} squadre caricate")
        for tid, c in self.centroids.items():
            print(f"  Team {tid}: LAB={c.round(1)}")

    def classify_player(self, frame: np.ndarray, bbox: tuple):
        if not self.centroids:
            return -1, 0.0
        color = _dominant_lab(frame, bbox)
        if color is None:
            return -1, 0.0
        dists   = {tid: float(np.linalg.norm(color - c))
                   for tid, c in self.centroids.items()}
        sorted_ = sorted(dists.items(), key=lambda x: x[1])
        best_id, best_d = sorted_[0]
        if len(sorted_) > 1:
            _, second_d = sorted_[1]
            if (second_d - best_d) < self.unknown_margin * second_d:
                return -1, best_d
        return best_id, best_d