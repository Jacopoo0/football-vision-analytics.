import cv2, numpy as np, supervision as sv, threading, time, warnings, requests
from collections import defaultdict, deque, Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from boxmot.trackers.botsort.botsort import BoTSORT
from team_classifier import TeamClassifier
from stats_tracker import StatsTracker
from homography import HomographyMapper
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ── Percorsi ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "raw" / "input_vid.mp4"
TEAM_JSON  = BASE_DIR / "team_colors.json"
MODEL_PT   = BASE_DIR / "models" / "soccana_best.pt"
OUTPUT_MP4 = BASE_DIR / "output_football_analysis.mp4"

# ── Config ────────────────────────────────────────────────────────────────────
MAX_SECONDS      = 60
PLAYER_CONF      = 0.18        # abbassato: recupera detection a bassa conf
PLAYER_IOU       = 0.50
TEAM_HISTORY_LEN = 30
INFER_SIZE       = 640         # aumentato: bbox più precise sui giocatori lontani
SAVE_VIDEO       = True
SHOW_PREVIEW     = False

FRAME_W = 960
FRAME_H = 540
PANEL_W = 360

CLS_PLAYER  = 0
CLS_BALL    = 1
CLS_REFEREE = 2

# ── Palette DAZN-style ────────────────────────────────────────────────────────
_BG         = (8,   10,  14)
_PANEL_BG   = (11,  14,  20)
_CARD_BG    = (17,  21,  30)
_BORDER     = (32,  40,  58)
_DIVIDER    = (24,  30,  44)
_WHITE      = (245, 247, 250)
_TEXT_MUTED = (110, 124, 150)
_TEXT_FAINT = (55,  66,  88)
_ACCENT     = (255, 75,  43)
_T0         = (60,  140, 255)
_T1         = (255, 72,  72)
_REF        = (200, 160, 255)
_BALL_C     = (255, 210, 50)

def _bgr(c): return (c[2], c[1], c[0])

T0_BGR   = _bgr(_T0)
T1_BGR   = _bgr(_T1)
REF_BGR  = _bgr(_REF)
BALL_BGR = _bgr(_BALL_C)
BG_BGR   = _bgr(_BG)

# ── CLAHE preprocessor ────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_sharpen_kernel = np.array([[0, -0.5, 0],
                             [-0.5, 3, -0.5],
                             [0, -0.5, 0]], dtype=np.float32)

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """CLAHE + leggero sharpening: migliora detection su video compressi."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    frame = cv2.filter2D(frame, -1, _sharpen_kernel)
    return frame

# ── Font cache ────────────────────────────────────────────────────────────────
_FONT_CACHE: dict = {}

def _load_font(size: int, bold=False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    urls = {
        True:  "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Bold.otf",
        False: "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Regular.otf",
    }
    cache_dir = BASE_DIR / ".font_cache"
    cache_dir.mkdir(exist_ok=True)
    fname = cache_dir / ("Inter-Bold.otf" if bold else "Inter-Regular.otf")
    if not fname.exists():
        try:
            r = requests.get(urls[bold], timeout=10)
            fname.write_bytes(r.content)
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
            return _FONT_CACHE[key]
    try:
        fnt = ImageFont.truetype(str(fname), size)
    except Exception:
        fnt = ImageFont.load_default()
    _FONT_CACHE[key] = fnt
    return fnt

def _preload_fonts():
    for sz in [8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48]:
        _load_font(sz, False)
        _load_font(sz, True)

# ── Pillow helpers ─────────────────────────────────────────────────────────────
def _t(d, text, xy, size, color=_WHITE, bold=False, anchor="lt"):
    d.text(xy, str(text), font=_load_font(size, bold), fill=color, anchor=anchor)

def _card(d, x1, y1, x2, y2, fill=_CARD_BG, border=_BORDER, radius=8):
    d.rounded_rectangle([x1, y1, x2, y2], radius=radius,
                        fill=fill, outline=border, width=1)

def _hbar_split(d, x, y, w, h, pct0, c0, c1, radius=3):
    d.rounded_rectangle([x, y, x+w, y+h], radius=radius, fill=_DIVIDER)
    split = max(radius*2, int(w * pct0 / 100))
    split = min(split, w - radius*2)
    d.rounded_rectangle([x,       y, x+split, y+h], radius=radius, fill=c0)
    d.rounded_rectangle([x+split, y, x+w,     y+h], radius=radius, fill=c1)

def _hbar_single(d, x, y, w, h, pct, color, radius=3):
    d.rounded_rectangle([x, y, x+w, y+h], radius=radius, fill=_DIVIDER)
    fw = max(radius*2, int(w * pct / 100))
    d.rounded_rectangle([x, y, x+fw, y+h], radius=radius, fill=color)

# ── Panel DAZN ────────────────────────────────────────────────────────────────
def build_panel(stats, sec, team_counts, frame_idx, max_frames) -> np.ndarray:
    W, H = PANEL_W, FRAME_H
    img  = Image.new("RGB", (W, H), _PANEL_BG)
    d    = ImageDraw.Draw(img)
    M    = 16

    # Header
    d.rectangle([0, 0, W, 52], fill=_BG)
    d.rectangle([0, 0, 3, 52], fill=_ACCENT)
    mins  = int(sec) // 60
    secs_ = int(sec) % 60
    _t(d, "MATCH ANALYSIS", (M, 10), 11, _TEXT_MUTED, bold=True)
    _t(d, f"{mins:02d}:{secs_:02d}", (W - M, 10), 28, _WHITE, bold=True, anchor="rt")
    d.rectangle([0, 48, W, 52], fill=_DIVIDER)
    prog_w = int(W * frame_idx / max(max_frames, 1))
    d.rectangle([0, 48, prog_w, 52], fill=_ACCENT)
    y = 60

    # Possesso
    _t(d, "BALL POSSESSION", (M, y), 9, _TEXT_MUTED, bold=True)
    y += 16
    p0, p1   = stats.possession_pct()
    rp0, rp1 = stats.recent_possession()
    cx = W // 2
    _t(d, f"{p0:.0f}%", (cx - 8, y), 40, _T0, bold=True, anchor="rt")
    _t(d, "–",           (cx,     y + 10), 14, _TEXT_FAINT, anchor="mt")
    _t(d, f"{p1:.0f}%", (cx + 8, y), 40, _T1, bold=True)
    y += 46
    _t(d, "TEAM 0", (M,     y), 8, _T0, bold=True)
    _t(d, "TEAM 1", (W - M, y), 8, _T1, bold=True, anchor="rt")
    y += 12
    _hbar_split(d, M, y, W - M*2, 7, p0, _T0, _T1, radius=3)
    y += 14
    _t(d, f"Ultimi 5s: {rp0:.0f}% – {rp1:.0f}%", (cx, y), 9, _TEXT_MUTED, anchor="mt")
    y += 18
    d.rectangle([M, y, W - M, y + 1], fill=_DIVIDER)
    y += 10

    # Giocatori in campo
    _t(d, "ON FIELD", (M, y), 9, _TEXT_MUTED, bold=True)
    y += 14
    hw = (W - M*2 - 8) // 2
    _card(d, M, y, M + hw, y + 50, fill=_CARD_BG, radius=6)
    d.rectangle([M, y, M + hw, y + 3], fill=_T0)
    _t(d, str(team_counts.get(0, 0)), (M + hw//2, y + 8),  28, _T0, bold=True, anchor="mt")
    _t(d, "TEAM 0",                    (M + hw//2, y + 38),  8, _TEXT_MUTED, anchor="mt")
    x2 = M + hw + 8
    _card(d, x2, y, x2 + hw, y + 50, fill=_CARD_BG, radius=6)
    d.rectangle([x2, y, x2 + hw, y + 3], fill=_T1)
    _t(d, str(team_counts.get(1, 0)), (x2 + hw//2, y + 8),  28, _T1, bold=True, anchor="mt")
    _t(d, "TEAM 1",                    (x2 + hw//2, y + 38),  8, _TEXT_MUTED, anchor="mt")
    y += 58
    d.rectangle([M, y, W - M, y + 1], fill=_DIVIDER)
    y += 10

    # Passaggi
    _t(d, "PASSES", (M, y), 9, _TEXT_MUTED, bold=True)
    y += 14
    ps0 = stats.passes.get(0, 0)
    ps1 = stats.passes.get(1, 0)
    total_ps = max(ps0 + ps1, 1)
    _card(d, M, y, W - M, y + 38, fill=_CARD_BG, radius=6)
    _t(d, str(ps0), (M + 20,     y + 10), 22, _T0, bold=True)
    _t(d, "T0",     (M + 20,     y + 32),  8, _TEXT_MUTED, anchor="lb")
    _t(d, str(ps1), (W - M - 20, y + 10), 22, _T1, bold=True, anchor="rt")
    _t(d, "T1",     (W - M - 20, y + 32),  8, _TEXT_MUTED, anchor="rb")
    bw = W - M*2 - 80
    bx = M + 40
    _hbar_split(d, bx, y + 20, bw, 5, ps0 / total_ps * 100, _T0, _T1, radius=2)
    y += 46
    d.rectangle([M, y, W - M, y + 1], fill=_DIVIDER)
    y += 10

    # Velocità
    _t(d, "SPEED  avg / max", (M, y), 9, _TEXT_MUTED, bold=True)
    y += 14
    s0, s1   = stats.avg_speed_kmh()
    mx0, mx1 = stats.max_speed_kmh()
    _card(d, M,  y, M + hw,     y + 52, fill=_CARD_BG, radius=6)
    d.rectangle([M, y, M + hw, y + 3], fill=_T0)
    _t(d, f"{s0:.1f}", (M + 10, y + 7),  22, _T0,        bold=True)
    _t(d, "km/h",       (M + 10, y + 30),  8, _TEXT_MUTED)
    _t(d, f"^{mx0:.1f}",(M + 10, y + 42),  9, _TEXT_FAINT)
    _card(d, x2, y, x2 + hw,    y + 52, fill=_CARD_BG, radius=6)
    d.rectangle([x2, y, x2 + hw, y + 3], fill=_T1)
    _t(d, f"{s1:.1f}", (x2 + 10, y + 7),  22, _T1,        bold=True)
    _t(d, "km/h",       (x2 + 10, y + 30),  8, _TEXT_MUTED)
    _t(d, f"^{mx1:.1f}",(x2 + 10, y + 42),  9, _TEXT_FAINT)
    y += 60
    d.rectangle([M, y, W - M, y + 1], fill=_DIVIDER)
    y += 10

    # Distanza
    _t(d, "DISTANCE", (M, y), 9, _TEXT_MUTED, bold=True)
    y += 14
    d0, d1 = stats.distance_meters()
    max_d  = max(d0, d1, 1.0)
    bw_full = W - M*2 - 70
    _t(d, "T0", (M, y + 4), 9, _T0, bold=True)
    _hbar_single(d, M + 22, y, bw_full, 9, d0 / max_d * 100, _T0, radius=3)
    _t(d, f"{d0/1000:.2f} km", (W - M, y + 4), 9, _T0, bold=True, anchor="rt")
    y += 18
    _t(d, "T1", (M, y + 4), 9, _T1, bold=True)
    _hbar_single(d, M + 22, y, bw_full, 9, d1 / max_d * 100, _T1, radius=3)
    _t(d, f"{d1/1000:.2f} km", (W - M, y + 4), 9, _T1, bold=True, anchor="rt")
    y += 22
    d.rectangle([M, y, W - M, y + 1], fill=_DIVIDER)
    y += 8

    # Footer
    refn = team_counts.get(2,  0)
    unkn = team_counts.get(-1, 0)
    _t(d, f"REF {refn}   UNK {unkn}", (M, y + 2), 9, _TEXT_FAINT)
    _t(d, f"{frame_idx} / {max_frames}", (W - M, y + 2), 8, _TEXT_FAINT, anchor="rt")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── Etichette giocatori ────────────────────────────────────────────────────────
def draw_player(frame, x1, y1, x2, y2, label, color):
    cx = (x1 + x2) // 2
    bw = max(8, int((x2 - x1) * 0.40))
    bh = max(3, int((x2 - x1) * 0.10))

    # Ombra ellittica sotto i piedi
    ov = frame.copy()
    cv2.ellipse(ov, (cx, y2), (bw, bh), 0, 0, 360, color, -1)
    cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
    cv2.ellipse(frame, (cx, y2), (bw, bh), 0, 0, 360, color, 1)

    # Label compatta
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.30
    thick      = 1
    (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thick)
    lx  = cx - lw // 2
    ly  = y1 - 4
    pad = 3
    cv2.rectangle(frame,
                  (lx - pad, ly - lh - pad),
                  (lx + lw + pad, ly + pad),
                  (6, 8, 14), -1)
    cv2.rectangle(frame,
                  (lx - pad, ly - lh - pad),
                  (lx + lw + pad, ly + pad),
                  color, 1)
    cv2.putText(frame, label, (lx, ly),
                font, font_scale, (240, 242, 246), thick, cv2.LINE_AA)


def draw_ball(frame, x1, y1, x2, y2):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = max(6, (x2 - x1) // 2)
    cv2.circle(frame, (cx, cy), r,    BALL_BGR,       -1)
    cv2.circle(frame, (cx, cy), r,    (180, 140, 20),  1)
    cv2.circle(frame, (cx, cy), r//3, (255, 255, 220), -1)


# ── Canvas compositing ────────────────────────────────────────────────────────
def build_canvas(frame_img, panel_img, minimap_img=None):
    fh, fw = frame_img.shape[:2]
    ph, pw = panel_img.shape[:2]
    total_h = max(fh, ph)
    total_w = fw + pw + 2
    C = np.full((total_h, total_w, 3), BG_BGR, dtype=np.uint8)
    fy = (total_h - fh) // 2
    C[fy:fy+fh, 0:fw] = frame_img
    py = (total_h - ph) // 2
    C[py:py+ph, fw+2:fw+2+pw] = panel_img
    C[:, fw:fw+2] = _bgr(_BORDER)

    if minimap_img is not None:
        mh, mw = minimap_img.shape[:2]
        mx = 10
        my = fy + fh - mh - 10
        if my >= fy and mx + mw <= fw:
            roi     = C[my:my+mh, mx:mx+mw].astype(np.float32)
            mm_f    = minimap_img.astype(np.float32)
            blended = (mm_f * 0.88 + roi * 0.12).astype(np.uint8)
            C[my:my+mh, mx:mx+mw] = blended
    return C


# ── Worker ────────────────────────────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    def __init__(self, model, team_classifier, tracker, stats, homography, fps):
        super().__init__(daemon=True)
        self.model      = model
        self.tc         = team_classifier
        self.tracker    = tracker
        self.stats      = stats
        self.homography = homography
        self.fps        = fps
        self.team_history      = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LEN))
        self._in_frame         = None
        self._in_lock          = threading.Lock()
        self._in_event         = threading.Event()
        self.result_frame      = None
        self.result_lock       = threading.Lock()
        self._stop             = False
        self._sec = self._fidx = self._maxf = 0
        self._last_ball_center = None
        self._ball_lost_frames = 0

    def submit(self, frame, sec, fidx, maxf):
        with self._in_lock:
            self._in_frame = frame.copy()
            self._sec  = sec
            self._fidx = fidx
            self._maxf = maxf
        self._in_event.set()

    def stop(self):
        self._stop = True
        self._in_event.set()

    def run(self):
        while not self._stop:
            self._in_event.wait()
            self._in_event.clear()
            if self._stop:
                break

            with self._in_lock:
                frame = self._in_frame.copy()
                sec, fidx, maxf = self._sec, self._fidx, self._maxf

            team_counts       = {0: 0, 1: 0, -1: 0, 2: 0}
            players_for_stats = []
            ball_center       = None
            field_positions   = {}

            try:
                # Preprocessing CLAHE + sharpening
                frame_proc = preprocess_frame(frame)

                self.homography.update_frame(frame_proc)

                results_p = self.model.predict(
                    frame_proc, imgsz=INFER_SIZE, conf=PLAYER_CONF,
                    iou=PLAYER_IOU, device="cuda", verbose=False,
                    classes=[CLS_PLAYER, CLS_REFEREE]
                )[0]
                results_b = self.model.predict(
                    frame_proc, imgsz=INFER_SIZE, conf=0.05,
                    iou=0.30, device="cuda", verbose=False,
                    classes=[CLS_BALL]
                )[0]

                player_det = sv.Detections.from_ultralytics(results_p)
                ball_det   = sv.Detections.from_ultralytics(results_b)

                if len(player_det) > 0:
                    areas = (
                        (player_det.xyxy[:, 2] - player_det.xyxy[:, 0]) *
                        (player_det.xyxy[:, 3] - player_det.xyxy[:, 1])
                    )
                    player_det = player_det[areas > 400]  # abbassato da 500

                if len(player_det) > 0:
                    dets_np = np.column_stack([
                        player_det.xyxy,
                        player_det.confidence,
                        player_det.class_id.astype(float)
                    ])
                    tracks = self.tracker.update(dets_np, frame_proc)
                else:
                    tracks = np.empty((0, 8))

            except Exception as e:
                print(f"Errore inference: {e}")
                tracks   = np.empty((0, 8))
                ball_det = sv.Detections.empty()

            # Palla
            if ball_det is not None and len(ball_det) > 0:
                best = int(np.argmax(ball_det.confidence))
                bx1, by1, bx2, by2 = map(int, ball_det.xyxy[best])
                ball_center = ((bx1+bx2)//2, (by1+by2)//2)
                self._last_ball_center = ball_center
                self._ball_lost_frames = 0
                draw_ball(frame, bx1, by1, bx2, by2)
                self.homography.update_ball(ball_center)
            else:
                self._ball_lost_frames += 1
                if self._ball_lost_frames <= 20 and self._last_ball_center:
                    ball_center = self._last_ball_center

            # Tracks
            for track in tracks:
                if len(track) < 7:
                    continue
                x1, y1, x2, y2 = map(int, track[:4])
                tid      = int(track[4])
                class_id = int(track[6])
                cx, cy   = (x1+x2)//2, (y1+y2)//2

                if class_id == CLS_REFEREE:
                    team_id = 2
                    label, color = "REF", REF_BGR
                else:
                    raw, _ = self.tc.classify_player(frame, (x1, y1, x2, y2))
                    if raw != -1:
                        self.team_history[tid].append(raw)
                    if len(self.team_history[tid]) >= 3:
                        team_id = Counter(self.team_history[tid]).most_common(1)[0][0]
                    else:
                        team_id = raw

                    if   team_id == 0: label, color = f"#{tid}", T0_BGR
                    elif team_id == 1: label, color = f"#{tid}", T1_BGR
                    else:              team_id = -1; label, color = f"#{tid}", _bgr((130, 144, 165))

                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                draw_player(frame, x1, y1, x2, y2, label, color)
                players_for_stats.append((cx, cy, team_id, tid))

                fp = self.homography.pixel_to_field((cx, cy))
                if fp is not None:
                    field_positions[tid] = (team_id, fp)

            self.stats.update(ball_center, players_for_stats)
            self.homography.update_players(field_positions)
            minimap = self.homography.render_minimap()

            frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
            panel       = build_panel(self.stats, sec, team_counts, fidx, maxf)
            canvas      = build_canvas(frame_small, panel, minimap)

            with self.result_lock:
                self.result_frame = canvas

    def get_canvas(self):
        with self.result_lock:
            return self.result_frame


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for path, label in [
        (VIDEO_PATH, "Video input"),
        (TEAM_JSON,  "team_colors.json"),
        (MODEL_PT,   "soccana_best.pt"),
    ]:
        if not path.exists():
            print(f"NON TROVATO: {label}\n  {path}")
            return

    print("Caricamento font...")
    _preload_fonts()

    print("Caricamento modelli...")
    model = YOLO(str(MODEL_PT))
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8),
                  imgsz=INFER_SIZE, device="cuda", verbose=False)

    team_classifier = TeamClassifier()
    team_classifier.load_samples(str(TEAM_JSON))

    reid_weights = BASE_DIR / "models" / "osnet_x0_25_msmt17.pt"

    # BoTSORT: migliore di StrongSORT su video calcistici con camera in movimento
    tracker = BoTSORT(
        reid_weights      = reid_weights,
        device            = "0",
        half              = False,
        track_high_thresh = 0.25,
        track_low_thresh  = 0.10,   # recupera detection a bassa confidenza
        new_track_thresh  = 0.20,
        track_buffer      = 120,    # frames prima di perdere un track
        match_thresh      = 0.85,
        cmc_method        = "sof",  # camera motion compensation
    )

    homography = HomographyMapper()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Impossibile aprire: {VIDEO_PATH}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(MAX_SECONDS * fps) if MAX_SECONDS else tot_frames
    spf        = 1.0 / fps

    stats  = StatsTracker(fps=fps)
    worker = InferenceWorker(model, team_classifier, tracker,
                             stats, homography, fps)
    worker.start()
    out = None

    if SHOW_PREVIEW:
        cv2.namedWindow("Football Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Football Analysis", FRAME_W + PANEL_W + 4, FRAME_H)

    print(f"Avvio  {max_frames} frame | Q / ESC per fermare")
    frame_idx   = 0
    last_canvas = None
    t_last = time.time()

    while True:
        if frame_idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        worker.submit(frame, sec, frame_idx, max_frames)
        canvas = worker.get_canvas()
        if canvas is not None:
            last_canvas = canvas

        if last_canvas is not None:
            if SAVE_VIDEO:
                if out is None:
                    h, w   = last_canvas.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out    = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, fps, (w, h))
                out.write(last_canvas)
            if SHOW_PREVIEW:
                cv2.imshow("Football Analysis", last_canvas)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break

        elapsed = time.time() - t_last
        if spf - elapsed > 0:
            time.sleep(spf - elapsed)
        t_last = time.time()

        if frame_idx % 60 == 0:
            p0, p1 = stats.possession_pct()
            d0, d1 = stats.distance_meters()
            s0, s1 = stats.avg_speed_kmh()
            print(f"[{frame_idx}/{max_frames}] {sec:.1f}s | "
                  f"Poss {p0:.0f}/{p1:.0f}% | "
                  f"Pass {stats.passes[0]}/{stats.passes[1]} | "
                  f"Dist {d0:.0f}/{d1:.0f}m | "
                  f"Vel {s0:.1f}/{s1:.1f} km/h")

    worker.stop()
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO:
        print(f"\nSalvato: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
