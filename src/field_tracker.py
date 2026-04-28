import cv2
import numpy as np
from ultralytics import YOLO

PITCH_KEYPOINTS_REAL = np.array([
    [0.0,    0.0],    #  0
    [0.0,   13.84],   #  1
    [52.5,  24.85],   #  2
    [0.0,   54.16],   #  3
    [16.5,  13.84],   #  4
    [16.5,  54.16],   #  5
    [0.0,   24.84],   #  6
    [5.5,   24.84],   #  7
    [0.0,   43.16],   #  8
    [5.5,   43.16],   #  9
    [0.0,   68.0],    # 10
    [52.5,   0.0],    # 11
    [52.5,  68.0],    # 12
    [52.5,  24.85],   # 13
    [52.5,  43.15],   # 14
    [52.5,  34.0],    # 15
    [105.0,  0.0],    # 16
    [105.0, 13.84],   # 17
    [88.5,  13.84],   # 18
    [105.0, 54.16],   # 19
    [88.5,  54.16],   # 20
    [105.0, 24.84],   # 21
    [99.5,  24.84],   # 22
    [105.0, 43.16],   # 23
    [99.5,  43.16],   # 24
    [105.0, 68.0],    # 25
    [84.85, 34.0],    # 26
    [43.35, 34.0],    # 27
    [61.65, 34.0],    # 28
], dtype=np.float32)

MIN_KEYPOINTS  = 4
KP_CONF_THRES  = 0.35
SMOOTH_ALPHA   = 0.75
MIN_SPREAD_PX  = 80


def _check_spread(pts: np.ndarray) -> bool:
    x_spread = pts[:, 0].max() - pts[:, 0].min()
    y_spread = pts[:, 1].max() - pts[:, 1].min()
    return x_spread >= MIN_SPREAD_PX or y_spread >= MIN_SPREAD_PX


class FieldTracker:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model       = YOLO(model_path)
        self.device      = device
        self.last_H      = None
        self.lost_frames = 0
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False)
        print("FieldTracker pronto.")

    def update(self, frame: np.ndarray) -> bool:
        results = self.model.predict(
            frame, imgsz=640, device=self.device,
            verbose=False, conf=KP_CONF_THRES
        )[0]

        if results.keypoints is None or len(results.keypoints.data) == 0:
            self.lost_frames += 1
            return False

        kps = results.keypoints.data[0].cpu().numpy()
        valid_mask = (
            (kps[:, 2] >= KP_CONF_THRES) &
            (kps[:, 0] > 1.0) &
            (kps[:, 1] > 1.0)
        )
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) < MIN_KEYPOINTS:
            self.lost_frames += 1
            return False

        src_pts = kps[valid_idx, :2].astype(np.float32)

        if not _check_spread(src_pts):
            self.lost_frames += 1
            return False

        dst_pts = PITCH_KEYPOINTS_REAL[valid_idx]
        H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            self.lost_frames += 1
            return False

        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if n_inliers < MIN_KEYPOINTS:
            self.lost_frames += 1
            return False

        if self.last_H is not None:
            H = SMOOTH_ALPHA * H + (1.0 - SMOOTH_ALPHA) * self.last_H

        self.last_H      = H
        self.lost_frames = 0
        return True

    def transform_points(self, points: np.ndarray):
        if self.last_H is None or points is None or len(points) == 0:
            return None
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.last_H)
        return out.reshape(-1, 2)

    @property
    def is_valid(self) -> bool:
        return self.last_H is not None

    @property
    def is_tracking(self) -> bool:
        return self.lost_frames == 0