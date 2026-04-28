import cv2
import numpy as np
from collections import deque

PITCH_LENGTH  = 105.0
PITCH_WIDTH   = 68.0
FIELD_COLOR   = (34, 85, 34)
LINE_COLOR    = (255, 255, 255)
BALL_TRAIL_LEN = 30


def create_minimap(scale: float = 6.0):
    w = int(PITCH_LENGTH * scale) + 80
    h = int(PITCH_WIDTH  * scale) + 80
    canvas = np.full((h, w, 3), FIELD_COLOR, dtype=np.uint8)

    def p(x, y):
        return int(x * scale) + 40, int(y * scale) + 40

    cv2.rectangle(canvas, p(0, 0), p(PITCH_LENGTH, PITCH_WIDTH), LINE_COLOR, 2)
    cv2.line(canvas, p(PITCH_LENGTH/2, 0), p(PITCH_LENGTH/2, PITCH_WIDTH), LINE_COLOR, 2)

    cx, cy = p(PITCH_LENGTH/2, PITCH_WIDTH/2)
    cv2.circle(canvas, (cx, cy), int(9.15 * scale), LINE_COLOR, 2)
    cv2.circle(canvas, (cx, cy), 3, LINE_COLOR, -1)

    bw = 40.32
    by1 = (PITCH_WIDTH - bw) / 2
    by2 = by1 + bw
    cv2.rectangle(canvas, p(0, by1),               p(16.5, by2),                LINE_COLOR, 2)
    cv2.rectangle(canvas, p(PITCH_LENGTH-16.5, by1), p(PITCH_LENGTH, by2),      LINE_COLOR, 2)

    sw = 18.32
    sy1 = (PITCH_WIDTH - sw) / 2
    sy2 = sy1 + sw
    cv2.rectangle(canvas, p(0, sy1),              p(5.5, sy2),              LINE_COLOR, 2)
    cv2.rectangle(canvas, p(PITCH_LENGTH-5.5, sy1), p(PITCH_LENGTH, sy2),  LINE_COLOR, 2)

    cv2.circle(canvas, p(11, PITCH_WIDTH/2),              3, LINE_COLOR, -1)
    cv2.circle(canvas, p(PITCH_LENGTH-11, PITCH_WIDTH/2), 3, LINE_COLOR, -1)

    return canvas, scale, (40, 40)


def field_to_minimap(x: float, y: float, scale: float, offset: tuple):
    return int(x * scale) + offset[0], int(y * scale) + offset[1]


def draw_player_on_minimap(minimap, x: int, y: int, track_id: int, color: tuple):
    cv2.circle(minimap, (x, y), 7, (0, 0, 0), -1)
    cv2.circle(minimap, (x, y), 5, color, -1)
    cv2.putText(minimap, str(track_id), (x + 7, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)


def draw_ball_trail(minimap, trail: deque):
    pts = list(trail)
    n   = len(pts)
    if n < 2:
        return
    for i, (px, py) in enumerate(pts):
        t      = i / (n - 1)
        radius = max(1, int(t * 5))
        b      = int(80  * (1 - t))
        g      = int(255 * t)
        r      = 255
        color  = (b, g, r)

        x1c = max(0, px - radius)
        y1c = max(0, py - radius)
        x2c = min(minimap.shape[1] - 1, px + radius)
        y2c = min(minimap.shape[0] - 1, py + radius)

        if x2c > x1c and y2c > y1c:
            roi  = minimap[y1c:y2c+1, x1c:x2c+1].astype(np.float32)
            mask = np.zeros_like(roi)
            cv2.circle(mask, (px - x1c, py - y1c), radius, color, -1)
            alpha = t * 0.85
            minimap[y1c:y2c+1, x1c:x2c+1] = np.clip(
                roi * (1 - alpha) + mask * alpha, 0, 255
            ).astype(np.uint8)


def draw_ball_on_minimap(minimap, x: int, y: int):
    cv2.circle(minimap, (x, y), 7, (0, 0, 0), -1)
    cv2.circle(minimap, (x, y), 5, (255, 255, 255), -1)


def draw_minimap_legend(minimap, legend_items: list):
    h, w = minimap.shape[:2]
    y    = h - 16
    cv2.rectangle(minimap, (0, y - 12), (w, h), (20, 20, 20), -1)
    for i, (color, label) in enumerate(legend_items):
        x = 16 + i * 88
        cv2.circle(minimap, (x, y + 2), 5, color, -1)
        cv2.putText(minimap, label, (x + 9, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1)