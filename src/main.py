import cv2
import numpy as np
import supervision as sv

from collections import defaultdict, deque, Counter
from pathlib import Path
from ultralytics import YOLO
from team_classifier import TeamClassifier
from field_tracker import FieldTracker
from minimap import (
    BALL_TRAIL_LEN,
    create_minimap,
    field_to_minimap,
    draw_player_on_minimap,
    draw_ball_on_minimap,
    draw_ball_trail,
    draw_minimap_legend,
)

BASE_DIR     = Path(__file__).resolve().parent.parent
VIDEO_PATH   = BASE_DIR / "data" / "raw" / "input_vid.mp4"
TEAM_JSON    = BASE_DIR / "team_colors.json"
MODEL_PT     = BASE_DIR / "models" / "soccana_best.pt"
MODEL_KP_PT  = BASE_DIR / "models" / "soccana_keypoint.pt"
OUTPUT_MP4   = BASE_DIR / "output_football_analysis.mp4"

MAX_SECONDS      = 120       # primi N secondi — metti None per tutto il video
PLAYER_CONF      = 0.30
PLAYER_IOU       = 0.45
TEAM_HISTORY_LEN = 10
INFER_SIZE       = 640
SHOW_PREVIEW     = False

FRAME_RENDER_W   = 480
FRAME_RENDER_H   = 270
MINIMAP_RENDER_H = 270

COLOR_BG     = (18, 18, 18)
COLOR_PANEL  = (28, 28, 28)
COLOR_ACCENT = (0, 200, 120)
COLOR_WHITE  = (235, 235, 235)
COLOR_MUTED  = (150, 150, 150)
COLOR_TEAM0  = (220, 80,  60)
COLOR_TEAM1  = (60,  180,  60)
COLOR_UNK    = (180, 180,  60)
COLOR_REF    = (200, 80,  220)
COLOR_BALL   = (255, 255, 255)

CLS_PLAYER  = 0
CLS_BALL    = 1
CLS_REFEREE = 2

LEGEND_ITEMS = [
    (COLOR_TEAM0, "TEAM 0"),
    (COLOR_TEAM1, "TEAM 1"),
    (COLOR_UNK,   "UNK"),
    (COLOR_REF,   "REF"),
    (COLOR_BALL,  "BALL"),
]


def draw_player_on_frame(frame, x1, y1, x2, y2, label, color):
    foot_x = (x1 + x2) // 2
    foot_y = y2
    rx     = max(10, int((x2 - x1) * 0.45))
    ry     = max(4,  int((x2 - x1) * 0.13))
    cv2.ellipse(frame, (foot_x, foot_y), (rx, ry), 0, 0, 360, color, 2)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    lx = x1
    ly = max(lh + 6, y1 - 4)
    cv2.rectangle(frame, (lx, ly - lh - 6), (lx + lw + 8, ly), color, -1)
    cv2.putText(frame, label, (lx + 4, ly - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, sec, team_counts, tracking_ok, frame_idx, total_frames):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 120), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    status_text  = "FIELD TRACK OK"  if tracking_ok else "FIELD TRACK LOST"
    status_color = COLOR_ACCENT       if tracking_ok else (0, 80, 255)
    progress     = (frame_idx / max(total_frames, 1)) * 100.0
    cv2.putText(frame, f"TIME {sec:.1f}s", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ACCENT, 2)
    cv2.putText(frame,
                f"T0:{team_counts.get(0,0)}  T1:{team_counts.get(1,0)}  "
                f"UNK:{team_counts.get(-1,0)}",
                (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_WHITE, 1)
    cv2.putText(frame, status_text, (14, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1)
    cv2.putText(frame, f"RENDER {progress:.1f}%", (14, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_MUTED, 1)


def build_dashboard(frame_img, minimap_img):
    fh, fw = frame_img.shape[:2]
    mh, mw = minimap_img.shape[:2]
    canvas = np.full((max(fh, mh) + 8, fw + mw + 12, 3), COLOR_BG, dtype=np.uint8)
    fy = (canvas.shape[0] - fh) // 2
    canvas[fy:fy+fh, 4:4+fw] = frame_img
    mx = fw + 8
    my = (canvas.shape[0] - mh) // 2
    canvas[my:my+mh, mx:mx+mw] = minimap_img
    cv2.rectangle(canvas, (3, fy-1),     (4+fw, fy+fh),   (55, 55, 55), 1)
    cv2.rectangle(canvas, (mx-1, my-1),  (mx+mw, my+mh),  (55, 55, 55), 1)
    return canvas


def main():
    for path, label in [
        (VIDEO_PATH,  "Video"),
        (TEAM_JSON,   "team_colors.json → esegui select_team_colors.py"),
        (MODEL_PT,    "soccana_best.pt  → scarica da HuggingFace"),
        (MODEL_KP_PT, "soccana_keypoint.pt → scarica da HuggingFace"),
    ]:
        if not path.exists():
            print(f"NON TROVATO: {label}\n  {path}")
            return

    print("Caricamento modelli...")
    model = YOLO(str(MODEL_PT))
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=INFER_SIZE, device="cuda", verbose=False)
    print(f"Classi: {model.names}")

    field_tracker   = FieldTracker(str(MODEL_KP_PT), device="cuda")
    team_classifier = TeamClassifier()
    team_classifier.load_samples(str(TEAM_JSON))
    team_history    = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LEN))
    track_last_team = {}
    tracker         = sv.ByteTrack()
    ball_trail      = deque(maxlen=BALL_TRAIL_LEN)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Impossibile aprire: {VIDEO_PATH}")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames   = int(MAX_SECONDS * fps) if MAX_SECONDS else total_frames

    minimap_base, mm_scale, mm_offset = create_minimap(scale=6.0)

    out       = None
    frame_idx = 0

    print(f"Rendering (max {MAX_SECONDS}s = {max_frames} frame)...")

    while True:
        if frame_idx >= max_frames:
            print(f"Raggiunto limite {MAX_SECONDS}s.")
            break

        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        sec         = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        minimap     = minimap_base.copy()
        team_counts = {0: 0, 1: 0, -1: 0}

        field_tracker.update(frame)

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

        # ── Giocatori + arbitri ───────────────────────────────────────────────
        if player_det is not None and len(player_det) > 0:
            for i, (box, track_id) in enumerate(
                zip(player_det.xyxy, player_det.tracker_id)
            ):
                x1, y1, x2, y2 = map(int, box)
                tid      = int(track_id) if track_id is not None else -1
                class_id = int(player_det.class_id[i]) \
                           if player_det.class_id is not None else CLS_PLAYER

                if class_id == CLS_REFEREE:
                    team_id      = -1
                    label, color = "REF", COLOR_REF
                else:
                    team_id_raw, _ = team_classifier.classify_player(
                        frame, (x1, y1, x2, y2)
                    )
                    if tid != -1 and team_id_raw != -1:
                        team_history[tid].append(team_id_raw)
                    if tid != -1 and len(team_history[tid]) >= 3:
                        team_id = Counter(team_history[tid]).most_common(1)[0][0]
                    else:
                        team_id = team_id_raw
                    if tid != -1:
                        track_last_team[tid] = team_id

                    if team_id == 0:
                        label, color = f"#{tid}", COLOR_TEAM0
                    elif team_id == 1:
                        label, color = f"#{tid}", COLOR_TEAM1
                    else:
                        label, color = f"#{tid}", COLOR_UNK

                team_counts[team_id] = team_counts.get(team_id, 0) + 1
                draw_player_on_frame(frame, x1, y1, x2, y2, label, color)

                if field_tracker.is_valid:
                    foot = np.array([[(x1+x2)/2, float(y2)]])
                    pt   = field_tracker.transform_points(foot)
                    if pt is not None:
                        fx, fy = pt[0]
                        px, py = field_to_minimap(fx, fy, mm_scale, mm_offset)
                        if 0 <= px < minimap.shape[1] and 0 <= py < minimap.shape[0]:
                            draw_player_on_minimap(minimap, px, py, tid, color)

        # ── Pallone ───────────────────────────────────────────────────────────
        if ball_det is not None and len(ball_det) > 0:
            best_idx           = int(np.argmax(ball_det.confidence))
            bx1, by1, bx2, by2 = map(int, ball_det.xyxy[best_idx])
            bcx, bcy           = (bx1+bx2)//2, (by1+by2)//2
            cv2.circle(frame, (bcx, bcy), 8, COLOR_BALL, 2)
            cv2.circle(frame, (bcx, bcy), 2, COLOR_BALL, -1)

            if field_tracker.is_valid:
                foot = np.array([[(bx1+bx2)/2, float(by2)]])
                pt   = field_tracker.transform_points(foot)
                if pt is not None:
                    fx, fy = pt[0]
                    px, py = field_to_minimap(fx, fy, mm_scale, mm_offset)
                    if 0 <= px < minimap.shape[1] and 0 <= py < minimap.shape[0]:
                        ball_trail.append((px, py))

        draw_ball_trail(minimap, ball_trail)
        if ball_trail:
            draw_ball_on_minimap(minimap, *ball_trail[-1])

        draw_hud(frame, sec, team_counts,
                 field_tracker.is_tracking, frame_idx, max_frames)
        draw_minimap_legend(minimap, LEGEND_ITEMS)

        frame_small   = cv2.resize(frame, (FRAME_RENDER_W, FRAME_RENDER_H))
        minimap_small = cv2.resize(
            minimap,
            (int(minimap.shape[1] * MINIMAP_RENDER_H / minimap.shape[0]),
             MINIMAP_RENDER_H)
        )
        dashboard = build_dashboard(frame_small, minimap_small)

        if out is None:
            h, w   = dashboard.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out    = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, fps, (w, h))
            if not out.isOpened():
                print("Errore VideoWriter")
                cap.release()
                return
            print(f"Writer: {w}x{h} @ {fps:.1f}fps")

        out.write(dashboard)

        if SHOW_PREVIEW:
            cv2.imshow("Dashboard", dashboard)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 30 == 0:
            pct = frame_idx / max_frames * 100
            print(f"{frame_idx}/{max_frames} | {pct:.1f}% | {sec:.1f}s | "
                  f"field={'OK' if field_tracker.is_tracking else 'LOST'} | "
                  f"trail={len(ball_trail)}")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f"\nSalvato: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()