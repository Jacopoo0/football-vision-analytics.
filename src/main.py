import cv2
import numpy as np
from ultralytics import YOLO
from minimap import create_minimap, draw_player_on_minimap, MINIMAP_WIDTH, MINIMAP_HEIGHT
from homography import build_homography_matrices, get_current_homography, project_point
from team_classifier import TeamClassifier
from collections import defaultdict, deque, Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "data" / "raw" / "input_vid.mp4"
MODEL_PATH = BASE_DIR / "yolov8n.pt"
TEAM_JSON = BASE_DIR / "team_colors.json"

CONF_THRESH = 0.35
IMG_SIZE = 640
TEAM_HISTORY_LEN = 12

# Colori dashboard
COLOR_BG        = (18, 18, 18)
COLOR_PANEL     = (30, 30, 30)
COLOR_ACCENT    = (0, 200, 120)
COLOR_WHITE     = (240, 240, 240)
COLOR_MUTED     = (140, 140, 140)
COLOR_TEAM0     = (220, 80, 60)
COLOR_TEAM1     = (60, 120, 220)
COLOR_UNK       = (180, 180, 60)


def resize_with_aspect_ratio(image, width=None, height=None):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is not None:
        new_width = width
        new_height = int(h * (width / w))
    else:
        new_height = height
        new_width = int(w * (height / h))
    return cv2.resize(image, (new_width, new_height))


def get_team_color_and_label(team_id):
    if team_id == 0:
        return COLOR_TEAM0, "TEAM 0"
    if team_id == 1:
        return COLOR_TEAM1, "TEAM 1"
    return COLOR_UNK, "UNK"


def draw_hud(frame, current_sec, segment, team_counts):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 115), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"TIME  {current_sec:.1f}s",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ACCENT, 2)
    cv2.putText(frame,
                f"SEG   {segment['start_sec']:.0f}s - {segment['end_sec']:.0f}s",
                (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_WHITE, 1)
    cv2.putText(frame,
                f"T0: {team_counts[0]}  T1: {team_counts[1]}  UNK: {team_counts[-1]}",
                (14, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_MUTED, 1)
    cv2.putText(frame, "SEMI-AUTO TEAM COLORS",
                (14, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_ACCENT, 1)


def draw_minimap_legend(minimap):
    legend_y = MINIMAP_HEIGHT - 22
    cv2.rectangle(minimap, (0, legend_y - 6), (MINIMAP_WIDTH, MINIMAP_HEIGHT), (20, 20, 20), -1)
    cv2.circle(minimap, (18, legend_y + 4), 5, COLOR_TEAM0, -1)
    cv2.putText(minimap, "TEAM 0", (28, legend_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1)
    cv2.circle(minimap, (108, legend_y + 4), 5, COLOR_TEAM1, -1)
    cv2.putText(minimap, "TEAM 1", (118, legend_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1)
    cv2.circle(minimap, (198, legend_y + 4), 5, COLOR_UNK, -1)
    cv2.putText(minimap, "UNK", (208, legend_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1)


def build_dashboard(frame_small, minimap_small):
    fh, fw = frame_small.shape[:2]
    mh, mw = minimap_small.shape[:2]

    total_w = fw + mw + 12
    total_h = max(fh, mh) + 8
    canvas = np.full((total_h, total_w, 3), COLOR_BG, dtype=np.uint8)

    fy = (total_h - fh) // 2
    canvas[fy:fy + fh, 4:4 + fw] = frame_small

    mx = fw + 8
    my = (total_h - mh) // 2
    canvas[my:my + mh, mx:mx + mw] = minimap_small

    cv2.rectangle(canvas, (3, fy - 1), (3 + fw + 1, fy + fh + 1), (60, 60, 60), 1)
    cv2.rectangle(canvas, (mx - 1, my - 1), (mx + mw + 1, my + mh + 1), (60, 60, 60), 1)

    return canvas


def main():
    print("VIDEO_PATH =", VIDEO_PATH)
    print("MODEL_PATH =", MODEL_PATH)
    print("TEAM_JSON  =", TEAM_JSON)

    if not VIDEO_PATH.exists():
        print(f"Video non trovato: {VIDEO_PATH}")
        return
    if not MODEL_PATH.exists():
        print(f"Modello non trovato: {MODEL_PATH}")
        return
    if not TEAM_JSON.exists():
        print(f"team_colors.json non trovato: {TEAM_JSON}")
        print("Esegui prima: python .\\src\\select_team_colors.py")
        return

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"Impossibile aprire il video: {VIDEO_PATH}")
        return

    homography_segments = build_homography_matrices()
    team_classifier = TeamClassifier()
    team_classifier.load_samples(str(TEAM_JSON))
    team_history = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LEN))

    cv2.namedWindow("Football Analysis Dashboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Football Analysis Dashboard", 1400, 600)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_height, frame_width = frame.shape[:2]
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        current_segment = get_current_homography(current_sec, homography_segments)
        H = current_segment["matrix"]

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONF_THRESH,
            imgsz=IMG_SIZE,
            verbose=False
        )

        result = results[0]
        minimap = create_minimap()
        team_counts = {0: 0, 1: 0, -1: 0}

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = (result.boxes.cls.int().cpu().tolist()
                         if result.boxes.cls is not None
                         else [0] * len(boxes))

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, box)
                box_w = x2 - x1
                box_h = y2 - y1

                if (box_w * box_h) > (frame_width * frame_height * 0.12):
                    continue
                if box_w / max(box_h, 1) > 1.2:
                    continue

                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2)

                team_id_raw, team_conf = team_classifier.classify_player(
                    frame, (x1, y1, x2, y2))

                if team_id_raw != -1:
                    team_history[int(track_id)].append(team_id_raw)

                if len(team_history[int(track_id)]) >= 3:
                    team_id = Counter(team_history[int(track_id)]).most_common(1)[0][0]
                else:
                    team_id = team_id_raw

                draw_color, draw_label = get_team_color_and_label(team_id)
                team_counts[team_id] = team_counts.get(team_id, 0) + 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.circle(frame, (foot_x, foot_y), 4, draw_color, -1)
                cv2.putText(
                    frame,
                    f"ID {track_id}  {draw_label}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    draw_color,
                    2
                )

                map_x, map_y = project_point(foot_x, foot_y, H)
                if 0 <= map_x < MINIMAP_WIDTH and 0 <= map_y < MINIMAP_HEIGHT:
                    draw_player_on_minimap(minimap, map_x, map_y, int(track_id), draw_color)

        draw_hud(frame, current_sec, current_segment, team_counts)
        draw_minimap_legend(minimap)

        frame_small = resize_with_aspect_ratio(frame, width=780)
        minimap_small = resize_with_aspect_ratio(minimap, height=frame_small.shape[0])

        dashboard = build_dashboard(frame_small, minimap_small)

        cv2.imshow("Football Analysis Dashboard", dashboard)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()