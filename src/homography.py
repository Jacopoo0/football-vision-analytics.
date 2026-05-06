import cv2
import numpy as np


FIELD_W = 105.0
FIELD_H = 68.0

MAP_W = 300
MAP_H = int(MAP_W * FIELD_H / FIELD_W)

# Palette minimap DAZN-style
_GRASS_DARK  = (22,  78,  38)
_GRASS_LIGHT = (28,  92,  46)
_LINE        = (220, 228, 240)
_LINE_DIM    = (140, 155, 175)
_T0          = (60,  140, 255)
_T1          = (255, 72,  72)
_REF         = (200, 160, 255)
_BALL_C      = (255, 210, 50)
_WHITE       = (220, 228, 240)


class HomographyMapper:
    def __init__(self):
        self.H            = None
        self.H_inv        = None
        self._calibrated  = False
        self._cal_every   = 30
        self._frame_cnt   = 0
        self._players     = {}
        self._ball_map    = None

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[2] = pts[np.argmax(s)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _detect_field_corners(self, frame):
        h, w = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(
            hsv, np.array([30, 40, 40]), np.array([90, 255, 255])
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < h * w * 0.15:
            return None

        peri   = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) == 4:
            return self._order_points(approx.reshape(4, 2).astype(np.float32))

        x, y, bw, bh = cv2.boundingRect(largest)
        mx, my = int(bw * 0.03), int(bh * 0.03)
        pts = np.array([
            [x+mx,    y+my   ],
            [x+bw-mx, y+my   ],
            [x+bw-mx, y+bh-my],
            [x+mx,    y+bh-my],
        ], dtype=np.float32)
        return self._order_points(pts)

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

    def update_frame(self, frame):
        self._frame_cnt += 1
        if self._frame_cnt % self._cal_every == 1:
            self.calibrate(frame)

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

    def render_minimap(self) -> np.ndarray:
        img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)

        # Strisce campo alternate
        stripe_w = MAP_W // 10
        for i in range(10):
            x0  = i * stripe_w
            x1  = min(x0 + stripe_w, MAP_W)
            col = _GRASS_LIGHT if i % 2 == 0 else _GRASS_DARK
            cv2.rectangle(img, (x0, 0), (x1, MAP_H), col, -1)

        lc = _LINE_DIM

        # Bordo campo
        cv2.rectangle(img, (1, 1), (MAP_W-2, MAP_H-2), lc, 1)

        # Metà campo
        cx = MAP_W // 2
        cv2.line(img, (cx, 1), (cx, MAP_H-2), lc, 1)

        # Cerchio centrale
        r = int(9.15 / FIELD_W * MAP_W)
        cv2.circle(img, (cx, MAP_H//2), r, lc, 1)
        cv2.circle(img, (cx, MAP_H//2), 2, lc, -1)
        cv2.circle(img, (cx, MAP_H//2), 1, _LINE, -1)

        # Aree di rigore
        pa_w = int(40.32 / FIELD_W * MAP_W)
        pa_h = int(16.5  / FIELD_H * MAP_H)
        ya0  = (MAP_H - pa_h) // 2
        cv2.rectangle(img, (1,          ya0), (pa_w,      ya0+pa_h), lc, 1)
        cv2.rectangle(img, (MAP_W-pa_w, ya0), (MAP_W-2,   ya0+pa_h), lc, 1)

        # Aree piccole
        sa_w = int(16.5  / FIELD_W * MAP_W)
        sa_h = int(18.32 / FIELD_H * MAP_H)
        ys0  = (MAP_H - sa_h) // 2
        cv2.rectangle(img, (1,          ys0), (sa_w,      ys0+sa_h), lc, 1)
        cv2.rectangle(img, (MAP_W-sa_w, ys0), (MAP_W-2,   ys0+sa_h), lc, 1)

        # Punti rigore
        pk_x = int(11.0 / FIELD_W * MAP_W)
        cv2.circle(img, (pk_x,       MAP_H//2), 1, lc, -1)
        cv2.circle(img, (MAP_W-pk_x, MAP_H//2), 1, lc, -1)

        # Giocatori
        for tid, (team_id, (mx, my)) in self._players.items():
            if   team_id == 0: col = _T0
            elif team_id == 1: col = _T1
            elif team_id == 2: col = _REF
            else:              col = (120, 130, 145)
            cv2.circle(img, (mx, my), 5, col,   -1)
            cv2.circle(img, (mx, my), 5, _WHITE, 1)

        # Palla
        if self._ball_map:
            bx, by = self._ball_map
            cv2.circle(img, (bx, by), 4, _BALL_C,        -1)
            cv2.circle(img, (bx, by), 4, (180, 140, 20),  1)
            cv2.circle(img, (bx, by), 1, (255, 255, 200), -1)

        # Overlay se non calibrato
        if not self._calibrated:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (MAP_W, MAP_H), (8, 10, 14), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.putText(img, "Calibrating...", (4, MAP_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, _LINE, 1, cv2.LINE_AA)

        return img