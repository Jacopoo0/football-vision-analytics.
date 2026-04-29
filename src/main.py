import cv2, numpy as np, supervision as sv, threading, time, warnings, requests
from io import BytesIO
from collections import defaultdict, deque, Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from team_classifier import TeamClassifier
from stats_tracker import StatsTracker
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
PLAYER_CONF      = 0.30
PLAYER_IOU       = 0.45
TEAM_HISTORY_LEN = 10
INFER_SIZE       = 480
SAVE_VIDEO       = True
SHOW_PREVIEW     = True

FRAME_W  = 960
FRAME_H  = 540
PANEL_W  = 380

CLS_PLAYER  = 0
CLS_BALL    = 1
CLS_REFEREE = 2

# ── Palette RGB ───────────────────────────────────────────────────────────────
_BG      = (10,  12,  18)
_PANEL   = (14,  18,  28)
_CARD    = (20,  26,  40)
_CARD2   = (26,  33,  50)
_BORDER  = (38,  48,  70)
_ACCENT  = (20,  200, 130)
_WHITE   = (248, 250, 252)
_MUTED   = ( 94, 108, 132)
_FAINT   = ( 30,  38,  58)
_T0      = ( 56, 132, 255)
_T1      = (248,  72,  72)
_REF     = (180, 120, 255)
_BALL    = (255, 200,  40)
_UNK     = (140, 155, 175)

def _bgr(c): return (c[2], c[1], c[0])
T0_BGR  = _bgr(_T0); T1_BGR = _bgr(_T1)
REF_BGR = _bgr(_REF); BALL_BGR = _bgr(_BALL)
UNK_BGR = _bgr(_UNK); ACC_BGR = _bgr(_ACCENT)
BG_BGR  = _bgr(_BG)

# ── Font (scarica Inter da Google Fonts) ──────────────────────────────────────
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
            print(f"Scarico font Inter {'Bold' if bold else 'Regular'}...")
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
    for sz in [10, 11, 12, 14, 18, 22, 28, 40, 52]:
        _load_font(sz, False)
        _load_font(sz, True)

# ── Pillow helpers ─────────────────────────────────────────────────────────────
def _t(d, text, xy, size, color=_WHITE, bold=False, anchor="lt"):
    d.text(xy, text, font=_load_font(size, bold), fill=color, anchor=anchor)

def _card(d, x1, y1, x2, y2, fill=_CARD, border=_BORDER, radius=10):
    d.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill,
                        outline=border, width=1)

def _hbar(d, x, y, w, h, pct, c_left, c_right, radius=4):
    d.rounded_rectangle([x, y, x+w, y+h], radius=radius, fill=_FAINT)
    split = max(radius*2, int(w * pct / 100))
    split = min(split, w - radius)
    d.rounded_rectangle([x, y, x+split, y+h], radius=radius, fill=c_left)
    if split < w - radius:
        d.rounded_rectangle([x+split, y, x+w, y+h], radius=radius, fill=c_right)

def _vline(d, x, y1, y2, color=_BORDER):
    d.line([(x, y1), (x, y2)], fill=color, width=1)

def _hline(d, x1, x2, y, color=_BORDER):
    d.line([(x1, y), (x2, y)], fill=color, width=1)

def _dot(d, cx, cy, r, color):
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)

# ── Panel ─────────────────────────────────────────────────────────────────────
def build_panel(stats, sec, team_counts, frame_idx, max_frames) -> np.ndarray:
    W, H = PANEL_W, FRAME_H
    img  = Image.new("RGB", (W, H), _PANEL)
    d    = ImageDraw.Draw(img)
    M    = 14   # margin

    # ────────────────── HEADER ────────────────────────────────────────────────
    d.rectangle([0, 0, W, 56], fill=_BG)
    # striscia accent
    d.rectangle([0, 0, 3, 56], fill=_ACCENT)
    # Titolo
    _t(d, "MATCH", (M+8, 10), 11, _MUTED)
    _t(d, "ANALYSIS", (M+8, 24), 18, _WHITE, bold=True)
    # Badge LIVE
    d.rounded_rectangle([M+8, 42, M+50, 54], radius=4, fill=_ACCENT)
    _t(d, "LIVE", (M+12, 43), 10, _BG, bold=True)
    # Tempo
    mins = int(sec)//60; secs_ = int(sec)%60
    _t(d, f"{mins:02d}:{secs_:02d}", (W-M, 12), 28, _WHITE, bold=True, anchor="rt")
    pct_s = f"{frame_idx/max(max_frames,1)*100:.0f}% completato"
    _t(d, pct_s, (W-M, 44), 10, _MUTED, anchor="rt")

    y = 66

    # ────────────────── POSSESSO ──────────────────────────────────────────────
    _card(d, M, y, W-M, y+110, radius=10)
    _t(d, "POSSESSO PALLA", (M+14, y+12), 10, _MUTED, bold=True)
    _hline(d, M+14, W-M-14, y+28, _BORDER)

    p0, p1   = stats.possession_pct()
    rp0, rp1 = stats.recent_possession()

    # Numeri grandi centrati
    cx0, cx1 = M+14 + (W//2 - M*2)//4, W - M - 14 - (W//2 - M*2)//4
    _t(d, f"{p0:.0f}", (M+14, y+35), 52, _T0, bold=True)
    _t(d, "%", (M+14, y+82), 14, _T0)
    _t(d, f"{p1:.0f}", (W-M-14, y+35), 52, _T1, bold=True, anchor="rt")
    _t(d, "%", (W-M-30, y+82), 14, _T1)

    # Barra
    _hbar(d, M+14, y+92, W-M*2-28, 12, p0, _T0, _T1, radius=6)
    _t(d, f"TEAM 0", (M+14, y+108), 9, _T0, bold=True)
    _t(d, f"TEAM 1", (W-M-14, y+108), 9, _T1, bold=True, anchor="rt")

    # Recente
    _t(d, f"▸ ultimi 5s  {rp0:.0f}%  –  {rp1:.0f}%",
       (W//2, y+98), 9, _MUTED, anchor="mt")

    y += 120

    # ────────────────── GIOCATORI + VELOCITÀ ──────────────────────────────────
    half = (W - M*3) // 2

    # Card T0
    _card(d, M, y, M+half, y+90, radius=10)
    d.rounded_rectangle([M, y, M+half, y+4], radius=2, fill=_T0)
    _dot(d, M+18, y+22, 4, _T0)
    _t(d, "TEAM 0", (M+26, y+15), 10, _T0, bold=True)
    t0n = team_counts.get(0, 0)
    _t(d, str(t0n), (M+14, y+32), 40, _WHITE, bold=True)
    _t(d, "in campo", (M+14, y+76), 9, _MUTED)

    # Card T1
    x2c = M*2 + half
    _card(d, x2c, y, x2c+half, y+90, radius=10)
    d.rounded_rectangle([x2c, y, x2c+half, y+4], radius=2, fill=_T1)
    _dot(d, x2c+18, y+22, 4, _T1)
    _t(d, "TEAM 1", (x2c+26, y+15), 10, _T1, bold=True)
    t1n = team_counts.get(1, 0)
    _t(d, str(t1n), (x2c+14, y+32), 40, _WHITE, bold=True)
    _t(d, "in campo", (x2c+14, y+76), 9, _MUTED)

    y += 100

    # ────────────────── VELOCITÀ MEDIA ────────────────────────────────────────
    _card(d, M, y, W-M, y+72, radius=10)
    _t(d, "VELOCITÀ MEDIA", (M+14, y+12), 10, _MUTED, bold=True)
    _hline(d, M+14, W-M-14, y+28, _BORDER)

    s0, s1 = stats.avg_speed_kmh()
    mid = W // 2
    _t(d, f"{s0:.1f}", (M+14, y+32), 28, _T0, bold=True)
    _t(d, "km/h", (M+14, y+60), 9, _MUTED)
    _vline(d, mid, y+32, y+68)
    _t(d, f"{s1:.1f}", (mid+14, y+32), 28, _T1, bold=True)
    _t(d, "km/h", (mid+14, y+60), 9, _MUTED)

    y += 82

    # ────────────────── DISTANZA ──────────────────────────────────────────────
    _card(d, M, y, W-M, y+86, radius=10)
    _t(d, "DISTANZA PERCORSA", (M+14, y+12), 10, _MUTED, bold=True)
    _hline(d, M+14, W-M-14, y+28, _BORDER)

    d0, d1 = stats.distance_meters()
    bw     = W - M*2 - 28
    max_d  = max(d0, d1, 1.0)

    bw0 = max(6, int(bw * d0/max_d))
    d.rounded_rectangle([M+14, y+34, M+14+bw, y+44], radius=4, fill=_FAINT)
    d.rounded_rectangle([M+14, y+34, M+14+bw0, y+44], radius=4, fill=_T0)
    _t(d, f"T0", (M+14, y+50), 9, _MUTED)
    _t(d, f"{d0/1000:.2f} km", (W-M-14, y+50), 10, _T0, bold=True, anchor="rt")

    bw1 = max(6, int(bw * d1/max_d))
    d.rounded_rectangle([M+14, y+62, M+14+bw, y+72], radius=4, fill=_FAINT)
    d.rounded_rectangle([M+14, y+62, M+14+bw1, y+72], radius=4, fill=_T1)
    _t(d, f"T1", (M+14, y+78), 9, _MUTED)
    _t(d, f"{d1/1000:.2f} km", (W-M-14, y+78), 10, _T1, bold=True, anchor="rt")

    y += 96

    # ────────────────── FOOTER ────────────────────────────────────────────────
    refn = team_counts.get(2, 0)
    unkn = team_counts.get(-1, 0)
    _t(d, f"Arbitri {refn}  ·  Non assegnati {unkn}",
       (M+14, y+4), 9, _MUTED)

    pct = frame_idx / max(max_frames, 1)
    d.rectangle([0, H-8, W, H], fill=_FAINT)
    d.rectangle([0, H-8, int(W*pct), H], fill=_ACCENT)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ── Disegno frame ──────────────────────────────────────────────────────────────
def draw_player(frame, x1, y1, x2, y2, label, c):
    fx, fy = (x1+x2)//2, y2
    rx = max(10, int((x2-x1)*0.45))
    ry = max(4,  int((x2-x1)*0.13))
    ov = frame.copy()
    cv2.ellipse(ov, (fx, fy), (rx, ry), 0, 0, 360, c, -1)
    cv2.addWeighted(ov, 0.20, frame, 0.80, 0, frame)
    cv2.ellipse(frame, (fx, fy), (rx, ry), 0, 0, 360, c, 2)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    lx = x1; ly = max(lh+6, y1-4)
    cv2.rectangle(frame, (lx, ly-lh-6), (lx+lw+8, ly), c, -1)
    cv2.putText(frame, label, (lx+4, ly-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1, cv2.LINE_AA)

def draw_ball(frame, x1, y1, x2, y2):
    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.circle(frame, (cx, cy), 9, BALL_BGR, -1)
    cv2.circle(frame, (cx, cy), 9, (0,150,200), 2)

def build_canvas(frame_img, panel_img):
    fh, fw = frame_img.shape[:2]
    ph, pw = panel_img.shape[:2]
    h = max(fh, ph)
    C = np.full((h+16, fw+pw+20, 3), BG_BGR, dtype=np.uint8)
    fy = (h-fh)//2 + 8
    C[fy:fy+fh, 8:8+fw] = frame_img
    px_ = fw+12; py_ = (h-ph)//2+8
    C[py_:py_+ph, px_:px_+pw] = panel_img
    cv2.rectangle(C, (7,fy-1),    (8+fw,fy+fh),   (38,48,70), 1)
    cv2.rectangle(C, (px_-1,py_-1),(px_+pw,py_+ph),(38,48,70), 1)
    return C

# ── Worker thread inference ────────────────────────────────────────────────────
class InferenceWorker(threading.Thread):
    def __init__(self, model, team_classifier, tracker, stats, fps):
        super().__init__(daemon=True)
        self.model          = model
        self.tc             = team_classifier
        self.tracker        = tracker
        self.stats          = stats
        self.fps            = fps
        self.team_history   = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LEN))

        self._in_frame  = None
        self._in_lock   = threading.Lock()
        self._in_event  = threading.Event()

        self.result_frame  = None
        self.result_panel  = None
        self.result_lock   = threading.Lock()

        self._stop = False
        self._sec  = 0.0
        self._fidx = 0
        self._maxf = 1
        self._team_counts = {0:0,1:0,-1:0,2:0}

    def submit(self, frame, sec, fidx, maxf):
        with self._in_lock:
            self._in_frame = frame.copy()
            self._sec = sec; self._fidx = fidx; self._maxf = maxf
        self._in_event.set()

    def stop(self): self._stop = True; self._in_event.set()

    def run(self):
        while not self._stop:
            self._in_event.wait()
            self._in_event.clear()
            if self._stop: break

            with self._in_lock:
                frame = self._in_frame.copy()
                sec, fidx, maxf = self._sec, self._fidx, self._maxf

            team_counts = {0:0,1:0,-1:0,2:0}
            players_for_stats = []
            ball_center = None

            try:
                results    = self.model.predict(frame, imgsz=INFER_SIZE,
                                                conf=PLAYER_CONF, iou=PLAYER_IOU,
                                                device="cuda", verbose=False)[0]
                all_det    = sv.Detections.from_ultralytics(results)
                ball_det   = all_det[all_det.class_id == CLS_BALL]
                player_det = all_det[all_det.class_id != CLS_BALL]
                player_det = self.tracker.update_with_detections(player_det)
            except Exception as e:
                print(f"Errore inference: {e}")
                player_det = sv.Detections.empty()
                ball_det   = sv.Detections.empty()

            if ball_det is not None and len(ball_det) > 0:
                best = int(np.argmax(ball_det.confidence))
                bx1,by1,bx2,by2 = map(int, ball_det.xyxy[best])
                ball_center = ((bx1+bx2)//2, (by1+by2)//2)
                draw_ball(frame, bx1, by1, bx2, by2)

            if player_det is not None and len(player_det) > 0:
                for i, (box, track_id) in enumerate(
                    zip(player_det.xyxy, player_det.tracker_id)
                ):
                    x1,y1,x2,y2 = map(int, box)
                    tid      = int(track_id) if track_id is not None else -1
                    class_id = int(player_det.class_id[i]) \
                               if player_det.class_id is not None else CLS_PLAYER
                    cx,cy    = (x1+x2)//2, (y1+y2)//2

                    if class_id == CLS_REFEREE:
                        team_id = 2; label,color = "REF", REF_BGR
                    else:
                        raw,_ = self.tc.classify_player(frame,(x1,y1,x2,y2))
                        if tid != -1 and raw != -1:
                            self.team_history[tid].append(raw)
                        if tid != -1 and len(self.team_history[tid]) >= 3:
                            team_id = Counter(self.team_history[tid]).most_common(1)[0][0]
                        else:
                            team_id = raw
                        if   team_id == 0: label,color = f"#{tid}", T0_BGR
                        elif team_id == 1: label,color = f"#{tid}", T1_BGR
                        else:              team_id=-1; label,color=f"#{tid}", UNK_BGR

                    team_counts[team_id] = team_counts.get(team_id,0)+1
                    draw_player(frame, x1,y1,x2,y2, label, color)
                    players_for_stats.append((cx,cy,team_id,tid))

            self.stats.update(ball_center, players_for_stats)
            self._team_counts = team_counts

            frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
            panel       = build_panel(self.stats, sec, team_counts, fidx, maxf)
            canvas      = build_canvas(frame_small, panel)

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
            print(f"NON TROVATO: {label}\n  {path}"); return

    print("Scarico/carico font...")
    _preload_fonts()

    print("Caricamento modelli...")
    model = YOLO(str(MODEL_PT))
    model.predict(np.zeros((640,640,3),dtype=np.uint8),
                  imgsz=INFER_SIZE, device="cuda", verbose=False)
    print(f"Classi: {model.names}")

    team_classifier = TeamClassifier()
    team_classifier.load_samples(str(TEAM_JSON))
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Impossibile aprire: {VIDEO_PATH}"); return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(MAX_SECONDS * fps) if MAX_SECONDS else tot_frames
    spf        = 1.0 / fps   # secondi per frame

    stats  = StatsTracker(fps=fps)
    worker = InferenceWorker(model, team_classifier, tracker, stats, fps)
    worker.start()

    out = None

    if SHOW_PREVIEW:
        cv2.namedWindow("Football Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Football Analysis", FRAME_W+PANEL_W+20, FRAME_H+16)

    print(f"▶  {max_frames} frame | Q/ESC per fermare")

    frame_idx   = 0
    last_canvas = None
    t_last      = time.time()

    while True:
        if frame_idx >= max_frames:
            print("Limite raggiunto."); break

        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Manda frame al worker (non blocca)
        worker.submit(frame, sec, frame_idx, max_frames)

        # Mostra l'ultimo canvas disponibile (anche se vecchio di 1 frame)
        canvas = worker.get_canvas()
        if canvas is not None:
            last_canvas = canvas

        if last_canvas is not None:
            if SAVE_VIDEO:
                if out is None:
                    h, w   = last_canvas.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out    = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, fps, (w, h))
                    print(f"Writer: {w}x{h} @ {fps:.1f}fps")
                out.write(last_canvas)

            if SHOW_PREVIEW:
                cv2.imshow("Football Analysis", last_canvas)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    print("Interrotto."); break

        # Rispetta il framerate del video originale
        elapsed = time.time() - t_last
        sleep_t = spf - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
        t_last = time.time()

        if frame_idx % 60 == 0:
            p0,p1 = stats.possession_pct()
            d0,d1 = stats.distance_meters()
            s0,s1 = stats.avg_speed_kmh()
            print(f"[{frame_idx}/{max_frames}] {sec:.1f}s | "
                  f"Poss {p0:.0f}/{p1:.0f}% | "
                  f"Dist {d0:.0f}/{d1:.0f}m | "
                  f"Vel {s0:.1f}/{s1:.1f}km/h")

    worker.stop()
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO: print(f"\nSalvato: {OUTPUT_MP4}")

if __name__ == "__main__":
    main()