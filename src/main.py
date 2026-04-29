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
INFER_SIZE       = 640
SAVE_VIDEO       = True       # True = salva anche mp4
SHOW_PREVIEW     = True       # True = finestra in tempo reale

FRAME_W  = 960
FRAME_H  = 540
PANEL_W  = 340

CLS_PLAYER  = 0
CLS_BALL    = 1
CLS_REFEREE = 2

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = (15,  17,  21)
PANEL_BG    = (22,  26,  32)
CARD_BG     = (30,  35,  43)
CARD_BG2    = (35,  41,  51)
ACCENT      = (0,   210, 130)
WHITE       = (240, 240, 240)
MUTED       = (120, 130, 145)
FAINT       = (55,  62,  74)
C_TEAM0     = (80,  130, 255)    # blu
C_TEAM1     = (255, 90,  70)     # rosso
C_REF       = (210, 90,  230)
C_BALL      = (255, 240, 80)
C_UNK       = (170, 170, 80)
DIVIDER     = (40,  46,  58)


# ═════════════════════════════════════════════════════════════════════════════
# Helper drawing
# ═════════════════════════════════════════════════════════════════════════════

def t(img, text, x, y, scale=0.52, color=WHITE, thickness=1, bold=False):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if bold else thickness, cv2.LINE_AA)


def rounded_rect(img, x1, y1, x2, y2, color, r=8, filled=True):
    """Rettangolo con angoli arrotondati (approssimato)."""
    th = -1 if filled else 1
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, th)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, th)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r),
                   (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, th)


def gradient_bar(img, x, y, w, h, pct, c_left, c_right, bg=(50,55,65)):
    """Barra bicolore con sfondo."""
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    split = int(w * pct / 100)
    if split > 0:
        cv2.rectangle(img, (x, y), (x+split, y+h), c_left, -1)
    if split < w:
        cv2.rectangle(img, (x+split, y), (x+w, y+h), c_right, -1)
    # highlight line on top
    cv2.line(img, (x, y), (x+w, y), (255,255,255,30), 1)


def draw_player(frame, x1, y1, x2, y2, label, color):
    fx, fy = (x1+x2)//2, y2
    rx = max(10, int((x2-x1)*0.45))
    ry = max(4,  int((x2-x1)*0.13))
    overlay = frame.copy()
    cv2.ellipse(overlay, (fx, fy), (rx, ry), 0, 0, 360, color, -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.ellipse(frame, (fx, fy), (rx, ry), 0, 0, 360, color, 2)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    lx = x1
    ly = max(lh+6, y1-4)
    rounded_rect(frame, lx, ly-lh-7, lx+lw+10, ly+1, color, r=4)
    cv2.putText(frame, label, (lx+5, ly-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)


def draw_ball(frame, x1, y1, x2, y2):
    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.circle(frame, (cx, cy), 9, C_BALL, -1)
    cv2.circle(frame, (cx, cy), 9, (200,180,0), 2)


# ═════════════════════════════════════════════════════════════════════════════
# Stats panel
# ═════════════════════════════════════════════════════════════════════════════

def build_panel(stats: StatsTracker, sec: float,
                team_counts: dict, frame_idx: int, max_frames: int) -> np.ndarray:

    P = np.full((FRAME_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    W = PANEL_W
    pad = 14

    # ── Header logo + tempo ───────────────────────────────────────────────────
    cv2.rectangle(P, (0, 0), (W, 60), (18, 22, 28), -1)
    # linea accent sinistra
    cv2.rectangle(P, (0, 0), (4, 60), ACCENT, -1)
    t(P, "FOOTBALL", pad+4, 24, 0.60, ACCENT, bold=True)
    t(P, "ANALYSIS", pad+4, 44, 0.46, MUTED)
    mins = int(sec)//60; secs_ = int(sec)%60
    t(P, f"{mins:02d}:{secs_:02d}", W-72, 30, 0.75, WHITE, bold=True)
    t(P, f"{frame_idx/max(max_frames,1)*100:.0f}%", W-52, 50, 0.42, MUTED)

    y = 72

    # ── Sezione: POSSESSO ─────────────────────────────────────────────────────
    rounded_rect(P, pad, y, W-pad, y+100, CARD_BG, r=6)
    t(P, "POSSESSO PALLA", pad+10, y+20, 0.46, MUTED)

    p0, p1 = stats.possession_pct()
    rp0, rp1 = stats.recent_possession()

    # Percentuali grandi
    t(P, f"{p0:.0f}%", pad+10,       y+56, 0.90, C_TEAM0, bold=True)
    t(P, f"{p1:.0f}%", W-pad-58,     y+56, 0.90, C_TEAM1, bold=True)
    t(P, "T0",         pad+10,        y+74, 0.38, MUTED)
    t(P, "T1",         W-pad-22,      y+74, 0.38, MUTED)

    # Barra possesso
    gradient_bar(P, pad+10, y+80, W-pad*2-20, 12, p0, C_TEAM0, C_TEAM1)

    # Recente (sotto barra)
    t(P, f"Ultimi 5s  T0:{rp0:.0f}%  T1:{rp1:.0f}%",
      pad+10, y+102, 0.36, MUTED)

    y += 108

    # ── Sezione: GIOCATORI ────────────────────────────────────────────────────
    y += 8
    rounded_rect(P, pad, y, W-pad, y+80, CARD_BG, r=6)
    t(P, "GIOCATORI IN CAMPO", pad+10, y+20, 0.46, MUTED)

    t0n = team_counts.get(0, 0)
    t1n = team_counts.get(1, 0)
    refn = team_counts.get(2, 0)
    unkn = team_counts.get(-1, 0)

    # Card T0
    cx0 = pad+10
    rounded_rect(P, cx0, y+28, cx0+80, y+70, CARD_BG2, r=5)
    cv2.rectangle(P, (cx0, y+28), (cx0+80, y+32), C_TEAM0, -1)
    t(P, f"{t0n}", cx0+28, y+60, 0.80, C_TEAM0, bold=True)
    t(P, "TEAM 0", cx0+6, y+44, 0.35, MUTED)

    # Card T1
    cx1 = W - pad - 90
    rounded_rect(P, cx1, y+28, cx1+80, y+70, CARD_BG2, r=5)
    cv2.rectangle(P, (cx1, y+28), (cx1+80, y+32), C_TEAM1, -1)
    t(P, f"{t1n}", cx1+28, y+60, 0.80, C_TEAM1, bold=True)
    t(P, "TEAM 1", cx1+6, y+44, 0.35, MUTED)

    t(P, f"Arb {refn}  Unk {unkn}", pad+10, y+80, 0.36, MUTED)

    y += 88

    # ── Sezione: DISTANZA ─────────────────────────────────────────────────────
    y += 8
    rounded_rect(P, pad, y, W-pad, y+90, CARD_BG, r=6)
    t(P, "DISTANZA PERCORSA", pad+10, y+20, 0.46, MUTED)

    d0, d1 = stats.distance_meters()
    max_d  = max(d0, d1, 1.0)
    bar_w  = W - pad*2 - 20

    # T0
    bw0 = int(bar_w * d0/max_d)
    cv2.rectangle(P, (pad+10, y+30), (pad+10+bw0, y+42), C_TEAM0, -1)
    cv2.rectangle(P, (pad+10, y+30), (pad+10+bar_w, y+42), FAINT, 1)
    t(P, f"T0   {d0/1000:.2f} km", pad+10, y+57, 0.44, C_TEAM0)

    # T1
    bw1 = int(bar_w * d1/max_d)
    cv2.rectangle(P, (pad+10, y+64), (pad+10+bw1, y+76), C_TEAM1, -1)
    cv2.rectangle(P, (pad+10, y+64), (pad+10+bar_w, y+76), FAINT, 1)
    t(P, f"T1   {d1/1000:.2f} km", pad+10, y+91, 0.44, C_TEAM1)

    y += 98

    # ── Footer progress ───────────────────────────────────────────────────────
    pct = frame_idx / max(max_frames, 1)
    # barra piena sfondo
    cv2.rectangle(P, (0, FRAME_H-8), (W, FRAME_H), FAINT, -1)
    # barra progresso
    cv2.rectangle(P, (0, FRAME_H-8), (int(W*pct), FRAME_H), ACCENT, -1)
    t(P, f"Frame {frame_idx}/{max_frames}", pad, FRAME_H-12, 0.36, MUTED)

    return P


# ═════════════════════════════════════════════════════════════════════════════
# Canvas
# ═════════════════════════════════════════════════════════════════════════════

def build_canvas(frame_img, panel_img):
    fh, fw = frame_img.shape[:2]
    ph, pw = panel_img.shape[:2]
    h = max(fh, ph)
    C = np.full((h+16, fw+pw+20, 3), BG, dtype=np.uint8)
    # Frame
    fy = (h-fh)//2 + 8
    C[fy:fy+fh, 8:8+fw] = frame_img
    rounded_rect(C, 7, fy-1, 8+fw, fy+fh, FAINT, r=4, filled=False)
    # Panel
    px_ = fw+12
    py_ = (h-ph)//2 + 8
    C[py_:py_+ph, px_:px_+pw] = panel_img
    rounded_rect(C, px_-1, py_-1, px_+pw, py_+ph, FAINT, r=4, filled=False)
    return C


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

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
    tracker         = sv.ByteTrack()
    stats           = StatsTracker()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Impossibile aprire: {VIDEO_PATH}"); return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(MAX_SECONDS * fps) if MAX_SECONDS else tot_frames
    out        = None
    frame_idx  = 0

    if SHOW_PREVIEW:
        cv2.namedWindow("Football Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Football Analysis", FRAME_W + PANEL_W + 20, FRAME_H + 16)

    print(f"Rendering (max {MAX_SECONDS}s = {max_frames} frame)...")

    while True:
        if frame_idx >= max_frames:
            print(f"\nLimite {MAX_SECONDS}s raggiunto."); break

        ok, frame = cap.read()
        if not ok:
            break

        frame_idx  += 1
        sec         = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        team_counts = {0: 0, 1: 0, -1: 0, 2: 0}
        players_for_stats = []
        ball_center = None

        # ── Inference ─────────────────────────────────────────────────────────
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
                    label, color = "REF", C_REF
                else:
                    raw, _ = team_classifier.classify_player(frame, (x1,y1,x2,y2))
                    if tid != -1 and raw != -1:
                        team_history[tid].append(raw)
                    if tid != -1 and len(team_history[tid]) >= 3:
                        team_id = Counter(team_history[tid]).most_common(1)[0][0]
                    else:
                        team_id = raw
                    if   team_id == 0: label, color = f"#{tid}", C_TEAM0
                    elif team_id == 1: label, color = f"#{tid}", C_TEAM1
                    else:              team_id = -1; label, color = f"#{tid}", C_UNK

                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                draw_player(frame, x1, y1, x2, y2, label, color)
                players_for_stats.append((cx, cy, team_id, tid))

        # ── Stats update ──────────────────────────────────────────────────────
        stats.update(ball_center, players_for_stats)

        # ── Composizione ──────────────────────────────────────────────────────
        frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
        panel       = build_panel(stats, sec, team_counts, frame_idx, max_frames)
        canvas      = build_canvas(frame_small, panel)

        # ── Output ────────────────────────────────────────────────────────────
        if SAVE_VIDEO:
            if out is None:
                h, w   = canvas.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out    = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, fps, (w, h))
                print(f"Writer: {w}x{h} @ {fps:.1f}fps")
            out.write(canvas)

        if SHOW_PREVIEW:
            cv2.imshow("Football Analysis", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # q o ESC per uscire
                print("\nInterrotto dall'utente.")
                break

        if frame_idx % 30 == 0:
            p0, p1 = stats.possession_pct()
            d0, d1 = stats.distance_meters()
            print(f"[{frame_idx}/{max_frames}] {frame_idx/max_frames*100:.1f}%"
                  f" | {sec:.1f}s | Poss T0:{p0:.0f}% T1:{p1:.0f}%"
                  f" | Dist T0:{d0:.0f}m T1:{d1:.0f}m")

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO:
        print(f"\nSalvato: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()