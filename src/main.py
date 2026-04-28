import cv2
import numpy as np
import supervision as sv
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from collections import defaultdict, deque, Counter
from pathlib import Path
from ultralytics import YOLO
from team_classifier import TeamClassifier
from stats_tracker import StatsTracker


BASE_DIR    = Path(__file__).resolve().parent.parent
VIDEO_PATH  = BASE_DIR / "data" / "raw" / "input_vid.mp4"
TEAM_JSON   = BASE_DIR / "team_colors.json"
MODEL_PT    = BASE_DIR / "models" / "soccana_best.pt"
OUTPUT_MP4  = BASE_DIR / "output_football_analysis.mp4"

MAX_SECONDS      = 60
PLAYER_CONF      = 0.30
PLAYER_IOU       = 0.45
TEAM_HISTORY_LEN = 10
INFER_SIZE       = 640
SHOW_PREVIEW     = False

FRAME_W  = 854
FRAME_H  = 480
PANEL_W  = 320

COLOR_BG     = (18,  18,  18)
COLOR_PANEL  = (28,  28,  28)
COLOR_PANEL2 = (38,  38,  38)
COLOR_ACCENT = (0,   200, 120)
COLOR_WHITE  = (235, 235, 235)
COLOR_MUTED  = (140, 140, 140)
COLOR_TEAM0  = (60,  80,  220)
COLOR_TEAM1  = (220, 80,   60)
COLOR_UNK    = (180, 180,  60)
COLOR_REF    = (200, 80,  220)
COLOR_BALL   = (255, 255, 255)

CLS_PLAYER  = 0
CLS_BALL    = 1
CLS_REFEREE = 2


# ── Disegno giocatore sul frame ───────────────────────────────────────────────
def draw_player(frame, x1, y1, x2, y2, label, color):
    fx, fy = (x1+x2)//2, y2
    rx = max(10, int((x2-x1)*0.45))
    ry = max(4,  int((x2-x1)*0.13))
    cv2.ellipse(frame, (fx, fy), (rx, ry), 0, 0, 360, color, 2)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    lx = x1
    ly = max(lh+6, y1-4)
    cv2.rectangle(frame, (lx, ly-lh-6), (lx+lw+8, ly), color, -1)
    cv2.putText(frame, label, (lx+4, ly-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)


def draw_ball(frame, x1, y1, x2, y2):
    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.circle(frame, (cx, cy), 8, COLOR_BALL, 2)
    cv2.circle(frame, (cx, cy), 2, COLOR_BALL, -1)


# ── Pannello statistiche ──────────────────────────────────────────────────────
def draw_stats_panel(stats: StatsTracker, sec: float,
                     team_counts: dict, frame_idx: int, max_frames: int):
    panel = np.full((FRAME_H, PANEL_W, 3), COLOR_PANEL, dtype=np.uint8)
    W = PANEL_W

    def txt(text, x, y, scale=0.55, color=COLOR_WHITE, bold=False):
        t = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, text, (x, y), t, scale, color,
                    2 if bold else 1, cv2.LINE_AA)

    def bar(x, y, w, h, pct0, c0, c1):
        cv2.rectangle(panel, (x, y), (x+w, y+h), (60,60,60), -1)
        w0 = int(w * pct0 / 100)
        cv2.rectangle(panel, (x, y), (x+w0, y+h), c0, -1)
        cv2.rectangle(panel, (x+w0, y), (x+w, y+h), c1, -1)

    def divider(y):
        cv2.line(panel, (12, y), (W-12, y), (55,55,55), 1)

    # ── Header ───────────────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (W, 50), COLOR_PANEL2, -1)
    mins  = int(sec) // 60
    secs_ = int(sec) % 60
    txt("FOOTBALL ANALYSIS", 12, 22, 0.52, COLOR_ACCENT, bold=True)
    txt(f"{mins:02d}:{secs_:02d}", 12, 44, 0.50, COLOR_MUTED)
    pct_str = f"{frame_idx/max(max_frames,1)*100:.0f}%"
    txt(pct_str, W-50, 44, 0.45, COLOR_MUTED)

    # ── Possesso ─────────────────────────────────────────────────────────────
    p0, p1 = stats.possession_pct()
    txt("POSSESSO PALLA", 12, 80, 0.52, COLOR_WHITE, bold=True)
    bar(12, 92, W-24, 18, p0, COLOR_TEAM0, COLOR_TEAM1)
    txt(f"{p0:.0f}%", 14, 126, 0.50, COLOR_TEAM0)
    txt(f"{p1:.0f}%", W-50, 126, 0.50, COLOR_TEAM1)
    divider(138)

    # ── Giocatori in campo ────────────────────────────────────────────────────
    t0_n = team_counts.get(0, 0)
    t1_n = team_counts.get(1, 0)
    txt("GIOCATORI IN CAMPO", 12, 162, 0.52, COLOR_WHITE, bold=True)

    # Team 0 box
    cv2.rectangle(panel, (12, 172), (W//2-6, 212), COLOR_PANEL2, -1)
    cv2.rectangle(panel, (12, 172), (W//2-6, 176), COLOR_TEAM0, -1)
    txt("TEAM 0", 20, 192, 0.45, COLOR_TEAM0)
    txt(str(t0_n), 20, 210, 0.70, COLOR_WHITE, bold=True)

    # Team 1 box
    cv2.rectangle(panel, (W//2+6, 172), (W-12, 212), COLOR_PANEL2, -1)
    cv2.rectangle(panel, (W//2+6, 172), (W-12, 176), COLOR_TEAM1, -1)
    txt("TEAM 1", W//2+14, 192, 0.45, COLOR_TEAM1)
    txt(str(t1_n), W//2+14, 210, 0.70, COLOR_WHITE, bold=True)

    divider(222)

    # ── Distanza percorsa ─────────────────────────────────────────────────────
    d0, d1 = stats.distance_meters()
    txt("DISTANZA PERCORSA", 12, 246, 0.52, COLOR_WHITE, bold=True)

    max_d = max(d0, d1, 1.0)

    # Barra T0
    bw = int((W-24) * d0 / max_d)
    cv2.rectangle(panel, (12, 256), (12+bw, 270), COLOR_TEAM0, -1)
    txt(f"T0  {d0/1000:.2f} km", 14, 282, 0.46, COLOR_TEAM0)

    # Barra T1
    bw1 = int((W-24) * d1 / max_d)
    cv2.rectangle(panel, (12, 292), (12+bw1, 306), COLOR_TEAM1, -1)
    txt(f"T1  {d1/1000:.2f} km", 14, 318, 0.46, COLOR_TEAM1)

    divider(328)

    # ── Arbitri / sconosciuti ─────────────────────────────────────────────────
    ref_n = team_counts.get(2, 0)
    unk_n = team_counts.get(-1, 0)
    txt(f"Arbitri: {ref_n}   Non assegnati: {unk_n}",
        12, 350, 0.42, COLOR_MUTED)

    # ── Footer progress bar ───────────────────────────────────────────────────
    pct = frame_idx / max(max_frames, 1)
    cv2.rectangle(panel, (0, FRAME_H-6), (W, FRAME_H), (50,50,50), -1)
    cv2.rectangle(panel, (0, FRAME_H-6),
                  (int(W*pct), FRAME_H), COLOR_ACCENT, -1)

    return panel


# ── Canvas finale ─────────────────────────────────────────────────────────────
def build_canvas(frame_img, panel_img):
    fh, fw = frame_img.shape[:2]
    ph, pw = panel_img.shape[:2]
    h = max(fh, ph)
    canvas = np.full((h+8, fw+pw+12, 3), COLOR_BG, dtype=np.uint8)
    fy = (h - fh) // 2 + 4
    canvas[fy:fy+fh, 4:4+fw] = frame_img
    px_ = fw + 8
    py_ = (h - ph) // 2 + 4
    canvas[py_:py_+ph, px_:px_+pw] = panel_img
    cv2.rectangle(canvas, (3, fy-1),    (4+fw, fy+fh),   (55,55,55), 1)
    cv2.rectangle(canvas, (px_-1, py_-1),(px_+pw, py_+ph),(55,55,55), 1)
    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    for path, label in [
        (VIDEO_PATH, "Video input"),
        (TEAM_JSON,  "team_colors.json → esegui select_team_colors.py"),
        (MODEL_PT,   "soccana_best.pt  → scarica da HuggingFace"),
    ]:
        if not path.exists():
            print(f"NON TROVATO: {label}\n  {path}")
            return

    print("Caricamento modelli...")
    model = YOLO(str(MODEL_PT))
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=INFER_SIZE, device="cuda", verbose=False)
    print(f"Classi: {model.names}")

    team_classifier = TeamClassifier()
    team_classifier.load_samples(str(TEAM_JSON))
    team_history    = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LEN))
    track_last_team = {}
    tracker         = sv.ByteTrack()
    stats           = StatsTracker()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Impossibile aprire: {VIDEO_PATH}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(MAX_SECONDS * fps) if MAX_SECONDS else tot_frames
    out        = None
    frame_idx  = 0

    print(f"Rendering (max {MAX_SECONDS}s = {max_frames} frame)...")

    while True:
        if frame_idx >= max_frames:
            print(f"Raggiunto limite {MAX_SECONDS}s.")
            break

        ok, frame = cap.read()
        if not ok:
            break

        frame_idx  += 1
        sec         = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        team_counts = {0: 0, 1: 0, -1: 0, 2: 0}
        players_for_stats = []
        ball_center = None

        try:
            results    = model.predict(
                frame, imgsz=INFER_SIZE, conf=PLAYER_CONF,
                iou=PLAYER_IOU, device="cuda", verbose=False
            )[0]
            all_det    = sv.Detections.from_ultralytics(results)
            ball_det   = all_det[all_det.class_id == CLS_BALL]
            player_det = all_det[all_det.class_id != CLS_BALL]
            player_det = tracker.update_with_detections(player_det)
        except Exception as e:
            print(f"Errore frame {frame_idx}: {e}")
            player_det = sv.Detections.empty()
            ball_det   = sv.Detections.empty()

        # ── Pallone ───────────────────────────────────────────────────────────
        if ball_det is not None and len(ball_det) > 0:
            best = int(np.argmax(ball_det.confidence))
            bx1, by1, bx2, by2 = map(int, ball_det.xyxy[best])
            ball_center = ((bx1+bx2)//2, (by1+by2)//2)
            draw_ball(frame, bx1, by1, bx2, by2)

        # ── Giocatori ─────────────────────────────────────────────────────────
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
                    label, color = "REF", COLOR_REF
                else:
                    raw, _ = team_classifier.classify_player(frame, (x1,y1,x2,y2))
                    if tid != -1 and raw != -1:
                        team_history[tid].append(raw)
                    if tid != -1 and len(team_history[tid]) >= 3:
                        team_id = Counter(team_history[tid]).most_common(1)[0][0]
                    else:
                        team_id = raw
                    if tid != -1:
                        track_last_team[tid] = team_id

                    if team_id == 0:
                        label, color = f"#{tid}", COLOR_TEAM0
                    elif team_id == 1:
                        label, color = f"#{tid}", COLOR_TEAM1
                    else:
                        team_id = -1
                        label, color = f"#{tid}", COLOR_UNK

                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                draw_player(frame, x1, y1, x2, y2, label, color)
                players_for_stats.append((cx, cy, team_id, tid))

        # ── Stats ─────────────────────────────────────────────────────────────
        stats.update(ball_center, players_for_stats)

        # ── Composizione canvas ───────────────────────────────────────────────
        frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
        panel       = draw_stats_panel(stats, sec, team_counts,
                                       frame_idx, max_frames)
        canvas      = build_canvas(frame_small, panel)

        if out is None:
            h, w   = canvas.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out    = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, fps, (w, h))
            if not out.isOpened():
                print("Errore VideoWriter")
                cap.release()
                return
            print(f"Writer: {w}x{h} @ {fps:.1f}fps")

        out.write(canvas)

        if SHOW_PREVIEW:
            cv2.imshow("Football Analysis", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 30 == 0:
            p0, p1 = stats.possession_pct()
            d0, d1 = stats.distance_meters()
            print(f"[{frame_idx}/{max_frames}] {frame_idx/max_frames*100:.1f}%"
                  f" | {sec:.1f}s | Poss T0:{p0:.0f}% T1:{p1:.0f}%"
                  f" | Dist T0:{d0:.0f}m T1:{d1:.0f}m")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print(f"\nSalvato: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()