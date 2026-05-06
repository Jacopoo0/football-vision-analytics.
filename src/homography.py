import cv2
import numpy as np

FIELD_W = 105.0
FIELD_H = 68.0
MAP_W   = 220
MAP_H   = int(MAP_W * FIELD_H / FIELD_W)

# Palette dark DAZN
_GRASS_D  = (19,  72,  34)
_GRASS_L  = (24,  88,  42)
_LINE     = (215, 224, 238)
_LINE_DIM = (130, 148, 172)
_T0       = ( 55, 135, 255)
_T1       = (255,  65,  65)
_REF      = (185, 145, 255)
_BALL_C   = (255, 210,  45)
_DOT_HL   = (238, 242, 248)
_BG_OVER  = (  6,   8,  12)
_BORDER   = ( 28,  36,  54)


def _field_to_map(fp):
    mx = int(fp[0] / FIELD_W * MAP_W)
    my = int(fp[1] / FIELD_H * MAP_H)
    return (int(np.clip(mx, 0, MAP_W-1)), int(np.clip(my, 0, MAP_H-1)))


class HomographyMapper:
    def __init__(self):
        self.H           = None
        self.H_inv       = None
        self._calibrated = False
        self._cal_every  = 30
        self._frame_cnt  = 0
        self._players    = {}
        self._ball_map   = None

    def _order_points(self, pts):
        rect = np.zeros((4,2), dtype=np.float32)
        s = pts.sum(axis=1); diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)];  rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        return rect

    def _detect_field_corners(self, frame):
        h, w = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([30,40,40]), np.array([90,255,255]))
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < h*w*0.12: return None
        peri   = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02*peri, True)
        if len(approx) == 4:
            return self._order_points(approx.reshape(4,2).astype(np.float32))
        x,y,bw,bh = cv2.boundingRect(largest)
        mx2,my2 = int(bw*0.03), int(bh*0.03)
        return self._order_points(np.array([
            [x+mx2,    y+my2   ],
            [x+bw-mx2, y+my2   ],
            [x+bw-mx2, y+bh-my2],
            [x+mx2,    y+bh-my2],
        ], dtype=np.float32))

    def calibrate(self, frame):
        src = self._detect_field_corners(frame)
        if src is None: return False
        dst = np.array([[0,0],[FIELD_W,0],[FIELD_W,FIELD_H],[0,FIELD_H]],
                       dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None: return False
        self.H = H; self.H_inv = np.linalg.inv(H)
        self._calibrated = True
        return True

    def update_frame(self, frame):
        self._frame_cnt += 1
        if self._frame_cnt % self._cal_every == 1:
            self.calibrate(frame)

    def pixel_to_field(self, px):
        if self.H is None: return None
        pt  = np.array([[[float(px[0]), float(px[1])]]], dtype=np.float32)
        fp  = cv2.perspectiveTransform(pt, self.H)[0][0]
        if -5 < fp[0] < FIELD_W+5 and -5 < fp[1] < FIELD_H+5:
            return (float(np.clip(fp[0],0,FIELD_W)),
                    float(np.clip(fp[1],0,FIELD_H)))
        return None

    def update_players(self, field_positions):
        self._players = {}
        for tid,(team_id,fp) in field_positions.items():
            self._players[tid] = (team_id, _field_to_map(fp))

    def update_ball(self, pixel_pt):
        fp = self.pixel_to_field(pixel_pt)
        self._ball_map = _field_to_map(fp) if fp else None

    def render_minimap(self) -> np.ndarray:
        img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)

        # ── Strisce campo alternate
        sw = MAP_W // 10
        for i in range(10):
            x0  = i * sw
            x1  = min(x0+sw, MAP_W)
            cv2.rectangle(img, (x0,0),(x1,MAP_H),
                          _GRASS_L if i%2==0 else _GRASS_D, -1)

        lc = _LINE_DIM

        # ── Bordo + metà campo
        cv2.rectangle(img, (1,1),(MAP_W-2,MAP_H-2), lc, 1)
        mx2 = MAP_W // 2
        cv2.line(img, (mx2,1),(mx2,MAP_H-2), lc, 1)

        # ── Cerchio centrocampo
        r  = int(9.15/FIELD_W * MAP_W)
        cy = MAP_H // 2
        cv2.circle(img, (mx2,cy), r, lc, 1)
        cv2.circle(img, (mx2,cy), 2, _LINE, -1)

        # ── Aree di rigore
        pa_w = int(40.32/FIELD_W * MAP_W)
        pa_h = int(16.5 /FIELD_H * MAP_H)
        ya0  = (MAP_H-pa_h)//2
        cv2.rectangle(img,(1,        ya0),(pa_w,    ya0+pa_h), lc, 1)
        cv2.rectangle(img,(MAP_W-pa_w,ya0),(MAP_W-2, ya0+pa_h), lc, 1)

        # ── Aree piccole
        sa_w = int(16.5 /FIELD_W * MAP_W)
        sa_h = int(18.32/FIELD_H * MAP_H)
        ys0  = (MAP_H-sa_h)//2
        cv2.rectangle(img,(1,         ys0),(sa_w,     ys0+sa_h), lc, 1)
        cv2.rectangle(img,(MAP_W-sa_w, ys0),(MAP_W-2,  ys0+sa_h), lc, 1)

        # ── Punti di rigore
        pk_x = int(11.0/FIELD_W * MAP_W)
        cv2.circle(img,(pk_x,       cy), 2, lc, -1)
        cv2.circle(img,(MAP_W-pk_x, cy), 2, lc, -1)

        # ── Semicerchi area di rigore
        arc_r = int(9.15/FIELD_W * MAP_W)
        cv2.ellipse(img,(pk_x,       cy), (arc_r,arc_r), 0, -60,  60,  lc, 1)
        cv2.ellipse(img,(MAP_W-pk_x, cy), (arc_r,arc_r), 0, 120, 240, lc, 1)

        # ── Porte
        gw = int(7.32/FIELD_H * MAP_H)
        gd = max(3, int(2.0/FIELD_W * MAP_W))
        gy0 = (MAP_H - gw) // 2
        cv2.rectangle(img,(0,    gy0),(gd,     gy0+gw), _LINE, 1)
        cv2.rectangle(img,(MAP_W-gd,gy0),(MAP_W,gy0+gw), _LINE, 1)

        # ── Giocatori
        for tid,(team_id,(px,py)) in self._players.items():
            if   team_id == 0: col = _T0
            elif team_id == 1: col = _T1
            elif team_id == 2: col = _REF
            else:              col = (100, 115, 140)
            cv2.circle(img,(px,py), 5, col,    -1)
            cv2.circle(img,(px,py), 5, _DOT_HL, 1)

        # ── Palla
        if self._ball_map:
            bx,by = self._ball_map
            cv2.circle(img,(bx,by), 4, _BALL_C,        -1)
            cv2.circle(img,(bx,by), 4, (160,130,15),    1)
            cv2.circle(img,(bx,by), 1, (255,250,200),  -1)

        # ── Label nome + overlay se non calibrato
        if not self._calibrated:
            ov = img.copy()
            cv2.rectangle(ov,(0,0),(MAP_W,MAP_H),_BG_OVER,-1)
            cv2.addWeighted(ov,0.50,img,0.50,0,img)
            cv2.putText(img,"Calibrating...",(4,MAP_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.30,_LINE,1,cv2.LINE_AA)
        else:
            cv2.putText(img,"MINIMAP",(4,9),
                        cv2.FONT_HERSHEY_SIMPLEX,0.25,_LINE_DIM,1,cv2.LINE_AA)

        return img