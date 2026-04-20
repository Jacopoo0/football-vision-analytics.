import cv2
import numpy as np

DEST_POINTS = np.float32([
    [20, 20],
    [480, 20],
    [20, 300],
    [480, 300]
])

HOMOGRAPHY_SEGMENTS = [
    {
        "start_sec": 0.0,
        "end_sec": 10.0,
        "source_points": np.float32([
            [2, 99],
            [842, 111],
            [6, 476],
            [842, 479]
        ])
    },
    {
        "start_sec": 10.0,
        "end_sec": 20.0,
        "source_points": np.float32([
            [3, 62],
            [496, 67],
            [3, 416],
            [844, 309]
        ])
    },
    {
        "start_sec": 20.0,
        "end_sec": 31.0,
        "source_points": np.float32([
            [3, 132],
            [846, 107],
            [5, 477],
            [844, 477]
        ])
    }
]


def build_homography_matrices():
    segments = []

    for segment in HOMOGRAPHY_SEGMENTS:
        H = cv2.getPerspectiveTransform(
            segment["source_points"],
            DEST_POINTS
        )

        segments.append({
            "start_sec": segment["start_sec"],
            "end_sec": segment["end_sec"],
            "matrix": H
        })

    return segments


def get_current_homography(current_sec, segments):
    for segment in segments:
        if segment["start_sec"] <= current_sec < segment["end_sec"]:
            return segment
    return segments[-1]


def project_point(x, y, matrix):
    point = np.array([[[x, y]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, matrix)
    map_x = int(projected[0][0][0])
    map_y = int(projected[0][0][1])
    return map_x, map_y