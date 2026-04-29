import cv2
import numpy as np
import supervision as sv
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from collections import defaultdict, deque, Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from team_classifier import TeamClassifier
from stats_tracker import StatsTracker

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
FRAME_SKIP       = 2
SAVE_VIDEO       = True
SHOW_PREVIEW     = True

FRAME_W  = 960
FRAME_H  = 540
PANEL_W  = 360

CLS_PLAYER  = 0
CLS_BALL    = 1
CLS_REFEREE = 2

# ── Palette (RGB per PIL, BGR per OpenCV) ─────────────────────────────────────
_BG       = (11,  14,  20)
_PANEL    = (17,  22,  32)
_CARD     = (24,  31,  45)
_CARD2    = (30,  38,  55)
_ACCENT   = (16,  185, 129)   # emerald
_WHITE    = (248, 250, 252)
_MUTED    = (100, 116, 139)
_FAINT    = (40,  50,  68)
_DIVIDER  = (35,  44,  62)
_T0       = (59,  130, 246)   # blue
_T1       = (239, 68,  68)    # red
_REF      = (192, 132, 252)   # purple
_BALL     = (251, 191, 36)    # amber
_UNK      = (148, 163, 184)

def _bgr(rgb): return (rgb[2], rgb[1], rgb[0])

BG_BGR    = _bgr(_BG)
PANEL_BGR = _bgr(_PANEL)
T0_BGR    = _bgr(_T0)
T1_BGR    = _bgr(_T1)
REF_BGR   = _bgr(_REF)
BALL_BGR  = _bgr(_BALL)
UNK_BGR   = _bgr(_UNK)
ACC_BGR   = _bgr(_ACCENT)


# ── Font loader ───────────────────────────────────────────────────────────────
def _font(size, bold=False):
    candidates = []
    if bold:
        candidates = ["arialbd.ttf", "Arial Bold.ttf",
                      "C:/Windows/Fonts/arialbd.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
    else:
        candidates = ["arial.ttf", "Arial.ttf",
                      "C:/Windows/Fonts/arial.ttf",
                      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
    for p in candidates:
        try: return ImageFont.truetype(p, size)
        except: pass
    return ImageFont.load_default()

# Pre-carica font
F_TITLE   = _font(13, bold=True)
F_LABEL   = _font(11)
F_LABEL_B = _font(11, bold=True)
F_NUM_XL  = _font(44, bold=True)
F_NUM_LG  = _font(26, bold=True)
F_NUM_MD  = _font(18, bold=True)
F_NUM_SM  = _font(14, bold=True)


# ── Pillow helpers ────────────────────────────────────────────────────────────
def _txt(d: ImageDraw, text, xy, font, color=_WHITE, anchor="lt"):
    d.text(xy, text, font=font, fill=color, anchor=anchor)


def _bar(d: ImageDraw, x, y, w, h, pct, c_left, c_right, bg=_FAINT, radius=3):
    # sfondo
    d.rounded_rectangle([x, y, x+w, y+h], radius=radius, fill=bg)
    # sinistra
    split = max(radius, int(w * pct / 100))
    split = min(split, w-radius)
    if split > 0:
        d.rounded_rectangle([x, y, x+split, y+h], radius=radius, fill=c_left)
    # destra
    if split < w:
        d.rounded_rectangle([x+split, y, x+w, y+h], radius=radius, fill=c_right)


def panel_to_cv(pil_img):
    """Converte immagine PIL RGB → numpy BGR per OpenCV."""
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ── Pannello statistiche (PIL) ────────────────────────────────────────────────
def build_panel(stats: StatsTracker, sec: float,
                team_counts: dict, frame_idx: int, max_frames: int) -> np.ndarray:
    W, H = PANEL_W, FRAME_H
    img = Image.new("RGB", (W, H), _PANEL)
    d   = ImageDraw.Draw(img)
    pad = 16

    # ── Header ────────────────────────────────────────────────────────────────
    d.rectangle([0, 0, W, 62], fill=(10, 13, 20))
    # striscia accent
    d.rectangle([0, 0, 4, 62], fill=_ACCENT)
    _txt(d, "FOOTBALL ANALYSIS", (pad+4, 14), F_TITLE, _WHITE)
    _txt(d, "LIVE STATS", (pad+4, 32), F_LABEL, _MUTED)
    mins = int(sec)//60; secs_ = int(sec)%60
    _txt(d, f"{mins:02d}:{secs_:02d}", (W-pad, 10), F_NUM_MD, _ACCENT, anchor="rt")
    pct_s = f"{frame_idx/max(max_frames,1)*100:.0f}%"
    _txt(d, pct_s, (W-pad, 32), F_LABEL, _MUTED, anchor="rt")

    y = 72

    # ── Possesso ──────────────────────────────────────────────────────────────
    d.rounded_rectangle([pad, y, W-pad, y+128], radius=8, fill=_CARD)
    _txt(d, "POSSESSO PALLA", (pad+12, y+10), F_LABEL, _MUTED)
    d.line([(pad, y+26), (W-pad, y+26)], fill=_DIVIDER, width=1)

    p0, p1 = stats.possession_pct()
    rp0, rp1 = stats.recent_possession()

    # Numeri grandi
    _txt(d, f"{p0:.0f}%", (pad+14, y+36), F_NUM_XL, _T0)
    _txt(d, f"{p1:.0f}%", (W-pad-14, y+36), F_NUM_XL, _T1, anchor="rt")
    _txt(d, "TEAM 0", (pad+14, y+84), F_LABEL_B, _T0)
    _txt(d, "TEAM 1", (W-pad-14, y+84), F_LABEL_B, _T1, anchor="rt")

    # Barra possesso
    _bar(d, pad+12, y+100, W-pad*2-24, 14, p0, _T0, _T1, radius=7)

    # Recente
    _txt(d, f"ultimi 5s  ·  {rp0:.0f}% – {rp1:.0f}%",
         (W//2, y+120), F_LABEL, _MUTED, anchor="mt")

    y += 138

    # ── In campo + Velocità ───────────────────────────────────────────────────
    half = (W - pad*3) // 2

    # Card T0 giocatori
    d.rounded_rectangle([pad, y, pad+half, y+88], radius=8, fill=_CARD)
    d.rectangle([pad, y, pad+half, y+4], fill=_T0)
    _txt(d, "IN CAMPO", (pad+10, y+12), F_LABEL, _MUTED)
    t0n = team_counts.get(0, 0)
    t1n = team_counts.get(1, 0)
    _txt(d, f"{t0n}", (pad+10, y+28), F_NUM_XL, _T0)
    _txt(d, "TEAM 0", (pad+10, y+76), F_LABEL, _MUTED)

    # Card T1 giocatori
    x2 = pad*2 + half
    d.rounded_rectangle([x2, y, x2+half, y+88], radius=8, fill=_CARD)
    d.rectangle([x2, y, x2+half, y+4], fill=_T1)
    _txt(d, "IN CAMPO", (x2+10, y+12), F_LABEL, _MUTED)
    _txt(d, f"{t1n}", (x2+10, y+28), F_NUM_XL, _T1)
    _txt(d, "TEAM 1", (x2+10, y+76), F_LABEL, _MUTED)

    y += 98

    # ── Velocità media ────────────────────────────────────────────────────────
    d.rounded_rectangle([pad, y, W-pad, y+70], radius=8, fill=_CARD)
    _txt(d, "VELOCITÀ MEDIA", (pad+12, y+10), F_LABEL, _MUTED)
    d.line([(pad, y+26), (W-pad, y+26)], fill=_DIVIDER, width=1)

    s0, s1 = stats.avg_speed_kmh()
    mid = W // 2
    _txt(d, f"{s0:.1f}", (pad+12, y+30), F_NUM_LG, _T0)
    _txt(d, "km/h", (pad+12, y+58), F_LABEL, _MUTED)
    _txt(d, f"{s1:.1f}", (mid+12, y+30), F_NUM_LG, _T1)
    _txt(d, "km/h", (mid+12, y+58), F_LABEL, _MUTED)
    d.line([(mid, y+30), (mid, y+64)], fill=_DIVIDER, width=1)

    y += 80

    # ── Distanza percorsa ─────────────────────────────────────────────────────
    d.rounded_rectangle([pad, y, W-pad, y+92], radius=8, fill=_CARD)
    _txt(d, "DISTANZA PERCORSA", (pad+12, y+10), F_LABEL, _MUTED)
    d.line([(pad, y+26), (W-pad, y+26)], fill=_DIVIDER, width=1)

    d0, d1 = stats.distance_meters()
    max_d  = max(d0, d1, 1.0)
    bw     = W - pad*2 - 24

    bw0 = max(6, int(bw * d0/max_d))
    d.rounded_rectangle([pad+12, y+32, pad+12+bw, y+44], radius=4, fill=_FAINT)
    d.rounded_rectangle([pad+12, y+32, pad+12+bw0, y+44], radius=4, fill=_T0)
    _txt(d, f"T0  ·  {d0/1000:.2f} km", (pad+12, y+50), F_LABEL_B, _T0)

    bw1 = max(6, int(bw * d1/max_d))
    d.rounded_rectangle([pad+12, y+64, pad+12+bw, y+76], radius=4, fill=_FAINT)
    d.rounded_rectangle([pad+12, y+64, pad+12+bw1, y+76], radius=4, fill=_T1)
    _txt(d, f"T1  ·  {d1/1000:.2f} km", (pad+12, y+82), F_LABEL_B, _T1)

    y += 102

    # ── Arbitri / Unknown ─────────────────────────────────────────────────────
    refn = team_counts.get(2, 0)
    unkn = team_counts.get(-1, 0)
    _txt(d, f"Arbitri: {refn}    Non assegnati: {unkn}",
         (pad+12, y+8), F_LABEL, _MUTED)

    # ── Progress bar ──────────────────────────────────────────────────────────
    pct = frame_idx / max(max_frames, 1)
    d.rectangle([0, H-10, W, H], fill=_FAINT)
    d.rectangle([0, H-10, int(W*pct), H], fill=_ACCENT)
    _txt(d, f"{frame_idx} / {max_frames} frames",
         (pad, H-22), F_LABEL, _MUTED)

    return panel_to_cv(img)


# ── Disegno frame ─────────────────────────────────────────────────────────────
def draw_player(frame, x1, y1, x2, y2, label, color_bgr):
    fx, fy = (x1+x2)//2, y2
    rx = max(10, int((x2-x1)*0.45))
    ry = max(4,  int((x2-x1)*0.13))
    ov = frame.copy()
    cv2.ellipse(ov, (fx, fy), (rx, ry), 0, 0, 360, color_bgr, -1)
    cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)
    cv2.ellipse(frame, (fx, fy), (rx, ry), 0, 0, 360, color_bgr, 2)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    lx = x1; ly = max(lh+6, y1-4)
    cv2.rectangle(frame, (lx, ly-lh-7), (lx+lw+10, ly+1), color_bgr, -1)
    cv2.putText(frame, label, (lx+5, ly-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)


def draw_ball(frame, x1, y1, x2, y2):
    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.circle(frame, (cx, cy), 9, BALL_BGR, -1)
    cv2.circle(frame, (cx, cy), 9, (0, 140, 200), 2)


# ── Canvas finale ─────────────────────────────────────────────────────────────
def build_canvas(frame_img, panel_img):
    fh, fw = frame_img.shape[:2]
    ph, pw = panel_img.shape[:2]
    h = max(fh, ph)
    C = np.full((h+16, fw+pw+20, 3), BG_BGR, dtype=np.uint8)
    fy = (h-fh)//2 + 8
    C[fy:fy+fh, 8:8+fw] = frame_img
    px_ = fw + 12
    py_ = (h-ph)//2 + 8
    C[py_:py_+ph, px_:px_+pw] = panel_img
    # Bordi sottili
    cv2.rectangle(C, (7, fy-1),    (8+fw, fy+fh),    (40,50,68), 1)
    cv2.rectangle(C, (px_-1, py_-1),(px_+pw, py_+ph), (40,50,68), 1)
    return C


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for path, label in [
        (VIDEO_PATH, "Video input"),
        (TEAM_JSON,  "team_colors.json"),
        (MODEL_PT,   "soccana_best.pt"),
    ]:
        if not path.exists():
            print(f"NON TROVATO: {label}\n  {path}"); return

    print("Caricamento modelli...")
    model = YOLO(str(MODEL_PT))
    model.predict(np.zeros((640,640,3),dtype=np.uint8),
                  imgsz=INFER_SIZE, device="cuda", verbose=False)
    print(f"Classi: {model.names}")

    team_classifier = TeamClassifier()
    team_classifier.load_samples(str(TEAM_JSON))
    team_history    = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LEN))
    tracker         = sv.ByteTrack()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Impossibile aprire: {VIDEO_PATH}"); return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(MAX_SECONDS * fps) if MAX_SECONDS else tot_frames
    stats      = StatsTracker(fps=fps)
    out        = None
    frame_idx  = 0

    if SHOW_PREVIEW:
        cv2.namedWindow("Football Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Football Analysis", FRAME_W+PANEL_W+20, FRAME_H+16)

    print(f"Rendering {max_frames} frame | Q/ESC per fermare")

    while True:
        if frame_idx >= max_frames:
            print("Limite raggiunto."); break

        ok, frame = cap.read()
        if not ok: break

        frame_idx += 1
        sec        = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if frame_idx % FRAME_SKIP != 0:
            continue

        team_counts       = {0:0, 1:0, -1:0, 2:0}
        players_for_stats = []
        ball_center       = None

        try:
            results    = model.predict(frame, imgsz=INFER_SIZE, conf=PLAYER_CONF,
                                       iou=PLAYER_IOU, device="cuda", verbose=False)[0]
            all_det    = sv.Detections.from_ultralytics(results)
            ball_det   = all_det[all_det.class_id == CLS_BALL]
            player_det = all_det[all_det.class_id != CLS_BALL]
            player_det = tracker.update_with_detections(player_det)
        except Exception as e:
            print(f"Errore frame {frame_idx}: {e}")
            player_det = sv.Detections.empty()
            ball_det   = sv.Detections.empty()

        # Pallone
        if ball_det is not None and len(ball_det) > 0:
            best = int(np.argmax(ball_det.confidence))
            bx1, by1, bx2, by2 = map(int, ball_det.xyxy[best])
            ball_center = ((bx1+bx2)//2, (by1+by2)//2)
            draw_ball(frame, bx1, by1, bx2, by2)

        # Giocatori
        if player_det is not None and len(player_det) > 0:
            for i, (box, track_id) in enumerate(
                zip(player_det.xyxy, player_det.tracker_id)
            ):
                x1, y1, x2, y2 = map(int, box)
                tid      = int(track_id) if track_id is not None else -1
                class_id = int(player_det.class_id[i]) \
                           if player_det.class_id is not None else CLS_PLAYER
                cx, cy   = (x1+x2)//2, (y1+y2)//2

                if class_id == CLS_REFEREE:
                    team_id      = 2
                    label, color = "REF", REF_BGR
                else:
                    raw, _ = team_classifier.classify_player(frame,(x1,y1,x2,y2))
                    if tid != -1 and raw != -1:
                        team_history[tid].append(raw)
                    if tid != -1 and len(team_history[tid]) >= 3:
                        team_id = Counter(team_history[tid]).most_common(1)[0][0]
                    else:
                        team_id = raw
                    if   team_id == 0: label, color = f"#{tid}", T0_BGR
                    elif team_id == 1: label, color = f"#{tid}", T1_BGR
                    else:              team_id=-1; label, color = f"#{tid}", UNK_BGR

                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                draw_player(frame, x1, y1, x2, y2, label, color)
                players_for_stats.append((cx, cy, team_id, tid))

        stats.update(ball_center, players_for_stats)

        frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
        panel       = build_panel(stats, sec, team_counts, frame_idx, max_frames)
        canvas      = build_canvas(frame_small, panel)

        if SAVE_VIDEO:
            if out is None:
                h, w   = canvas.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out    = cv2.VideoWriter(str(OUTPUT_MP4), fourcc,
                                         fps/FRAME_SKIP, (w, h))
                print(f"Writer: {w}x{h}")
            out.write(canvas)

        if SHOW_PREVIEW:
            cv2.imshow("Football Analysis", canvas)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27): print("Interrotto."); break

        if frame_idx % 60 == 0:
            p0, p1 = stats.possession_pct()
            d0, d1 = stats.distance_meters()
            s0, s1 = stats.avg_speed_kmh()
            print(f"[{frame_idx}/{max_frames}] {sec:.1f}s | "
                  f"Poss {p0:.0f}/{p1:.0f}% | "
                  f"Dist {d0:.0f}/{d1:.0f}m | "
                  f"Vel {s0:.1f}/{s1:.1f}km/h")

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO: print(f"\nSalvato: {OUTPUT_MP4}")

if __name__ == "__main__":
    main()