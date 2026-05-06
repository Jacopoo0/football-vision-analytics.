import cv2
import numpy as np


FIELD_W = 105.0
FIELD_H = 68.0

MAP_W = 320
MAP_H = int(MAP_W * FIELD_H / FIELD_W)

_GREEN      = (34,  139,  34)
_WHITE      = (255, 255, 255)
_T0         = (255, 132,  56)
_T1         = (72,   72, 248)
_BALL_COLOR = (40,  200, 255)


class HomographyMapper:
    def __init__(self):
        self.H            = None
        self.H_inv        = None
        self._calibrated  = False
        self._cal_every   = 30
        self._frame_cnt   = 0
        self._field_pts   = np.array([
            [0,       0      ],
            [FIELD_W, 0      ],
            [FIELD_W, FIELD_H],
            [0,       FIELD_H],
        ], dtype=np.float32)
        self._players  = {}
        self._ball_map = None

    # ── ordina 4 punti: TL, TR, BR, BL ───────────────────────────────────────
    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]     # top-left
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[2] = pts[np.argmax(s)]     # bottom-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    # ── rileva i 4 angoli del campo ───────────────────────────────────────────
    def _detect_field_corners(self, frame):
        h, w = frame.shape[:2]

        # 1) Maschera verde con pulizia morfologica
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(
            hsv,
            np.array([30, 40, 40]),
            np.array([90, 255, 255])
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  kernel)

        # 2) Contorno verde piu' grande
        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)

        # Il campo deve occupare almeno il 15% del frame
        if area < h * w * 0.15:
            return None

        # 3) Approssimazione poligonale: se otteniamo 4 punti usiamoli
        peri   = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            return self._order_points(pts)

        # 4) Fallback: bounding rect con margine per escludere tribune
        x, y, bw, bh = cv2.boundingRect(largest)
        margin_x = int(bw * 0.03)
        margin_y = int(bh * 0.03)
        x  += margin_x
        y  += margin_y
        bw -= margin_x * 2
        bh -= margin_y * 2
        pts = np.array([
            [x,      y      ],
            [x + bw, y      ],
            [x + bw, y + bh ],
            [x,      y + bh ],
        ], dtype=np.float32)
        return self._order_points(pts)

    # ── calcola la matrice di omografia ───────────────────────────────────────
    def calibrate(self, frame):
        src_pts = self._detect_field_corners(frame)
        if src_pts is None:
            return False

        dst_pts = np.array([
            [0,       0      ],
            [FIELD_W, 0      ],
            [FIELD_W, FIELD_H],
            [0,       FIELD_H],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return False

        self.H           = H
        self.H_inv       = np.linalg.inv(H)
        self._calibrated = True
        return True

    # ── chiamato ogni frame da run() ──────────────────────────────────────────
    def update_frame(self, frame):
        self._frame_cnt += 1
        if self._frame_cnt % self._cal_every == 1:
            self.calibrate(frame)

    # ── pixel frame → coordinate metriche campo ───────────────────────────────
    def pixel_to_field(self, pixel_pt):
        if self.H is None:
            return None
        px = np.array([[[float(pixel_pt[0]), float(pixel_pt[1])]]], dtype=np.float32)
        fp = cv2.perspectiveTransform(px, self.H)[0][0]
        if -5 < fp[0] < FIELD_W + 5 and -5 < fp[1] < FIELD_H + 5:
            return (
                float(np.clip(fp[0], 0, FIELD_W)),
                float(np.clip(fp[1], 0, FIELD_H))
            )
        return None

    # ── coordinate metriche campo → pixel minimap ─────────────────────────────
    def field_to_map(self, field_pt):
        mx = int(field_pt[0] / FIELD_W * MAP_W)
        my = int(field_pt[1] / FIELD_H * MAP_H)
        return (
            int(np.clip(mx, 0, MAP_W - 1)),
            int(np.clip(my, 0, MAP_H - 1))
        )

    def update_players(self, field_positions):
        self._players = {}
        for tid, (team_id, fp) in field_positions.items():
            self._players[tid] = (team_id, self.field_to_map(fp))

    def update_ball(self, pixel_pt):
        fp = self.pixel_to_field(pixel_pt)
        self._ball_map = self.field_to_map(fp) if fp else None

    # ── rendering minimap ─────────────────────────────────────────────────────
    def render_minimap(self) -> np.ndarray:
        img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
        img[:] = _GREEN

        # Bordo campo
        cv2.rectangle(img, (0, 0), (MAP_W - 1, MAP_H - 1), _WHITE, 1)

        # Metà campo
        cx = MAP_W // 2
        cv2.line(img, (cx, 0), (cx, MAP_H), _WHITE, 1)

        # Cerchio centrale
        r = int(9.15 / FIELD_W * MAP_W)
        cv2.circle(img, (cx, MAP_H // 2), r, _WHITE, 1)
        cv2.circle(img, (cx, MAP_H // 2), 2, _WHITE, -1)

        # Aree di rigore
        pa_w = int(40.32 / FIELD_W * MAP_W)
        pa_h = int(16.5  / FIELD_H * MAP_H)
        y0   = (MAP_H - pa_h) // 2
        cv2.rectangle(img, (0,          y0), (pa_w,        y0 + pa_h), _WHITE, 1)
        cv2.rectangle(img, (MAP_W-pa_w, y0), (MAP_W - 1,   y0 + pa_h), _WHITE, 1)

        # Giocatori
        for tid, (team_id, (mx, my)) in self._players.items():
            color = _T0 if team_id == 0 else (_T1 if team_id == 1 else (200, 100, 255))
            cv2.circle(img, (mx, my), 5, color,  -1)
            cv2.circle(img, (mx, my), 5, _WHITE,  1)

        # Palla
        if self._ball_map:
            bx, by = self._ball_map
            cv2.circle(img, (bx, by), 4, _BALL_COLOR, -1)
            cv2.circle(img, (bx, by), 4, _WHITE,       1)

        # Bordo grigio esterno
        cv2.rectangle(img, (0, 0), (MAP_W - 1, MAP_H - 1), (80, 80, 80), 1)

        # Label calibrazione
        if not self._calibrated:
            cv2.putText(img, "Calibrazione...", (4, MAP_H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, _WHITE, 1)

        return img
