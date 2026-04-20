import cv2
from ultralytics import YOLO
from minimap import create_minimap, draw_player_on_minimap
from homography import build_homography_matrices, get_current_homography, project_point
from team_classifier import TeamColorClassifier

video_path = "data/raw/input_vid.mp4"
prototype_path = "team_colors.json"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(video_path)
homography_segments = build_homography_matrices()

team_classifier = TeamColorClassifier(
    history_size=10,
    other_absolute_threshold=40.0,
    ambiguity_margin=8.0
)
team_classifier.load_prototypes(prototype_path)


def resize_with_aspect_ratio(image, width=None, height=None):
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None:
        scale = width / w
        new_width = width
        new_height = int(h * scale)
    else:
        scale = height / h
        new_height = height
        new_width = int(w * scale)

    return cv2.resize(image, (new_width, new_height))


def draw_info(frame, current_sec, segment):
    cv2.putText(
        frame,
        f"TIME: {current_sec:.1f}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"SEGMENT: {segment['start_sec']:.0f}-{segment['end_sec']:.0f}s",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SEMI-AUTO TEAM COLORS",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


cv2.namedWindow("Football Analysis Dashboard", cv2.WINDOW_NORMAL)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_height, frame_width = frame.shape[:2]
    current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    current_segment = get_current_homography(current_sec, homography_segments)
    H = current_segment["matrix"]

    result = model.track(
        frame,
        persist=True,
        conf=0.35,
        imgsz=640
    )[0]

    minimap = create_minimap()

    if result.boxes is not None and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        if result.boxes.cls is not None:
            class_ids = result.boxes.cls.int().cpu().tolist()
        else:
            class_ids = [0] * len(boxes)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if class_id != 0:
                continue

            x1, y1, x2, y2 = box

            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            frame_area = frame_width * frame_height

            if box_area > frame_area * 0.12:
                continue

            if box_w / max(box_h, 1) > 1.2:
                continue

            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            jersey_color = team_classifier.extract_jersey_color(frame, box)
            team_id = team_classifier.predict_team(int(track_id), jersey_color)

            draw_color = team_classifier.get_team_color(team_id)
            draw_label = team_classifier.get_team_label(team_id)

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            cv2.circle(frame, (foot_x, foot_y), 4, (255, 255, 255), -1)

            cv2.putText(
                frame,
                f"ID {int(track_id)} {draw_label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                draw_color,
                2
            )

            map_x, map_y = project_point(foot_x, foot_y, H)

            if 0 <= map_x < 500 and 0 <= map_y < 320:
                draw_player_on_minimap(minimap, map_x, map_y, int(track_id), draw_color)

    draw_info(frame, current_sec, current_segment)

    cv2.putText(
        minimap,
        "BLUE/RED = TEAMS | YELLOW = OTHERS",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    frame_small = resize_with_aspect_ratio(frame, width=700)
    minimap_small = resize_with_aspect_ratio(minimap, height=frame_small.shape[0])

    dashboard = cv2.hconcat([frame_small, minimap_small])

    cv2.imshow("Football Analysis Dashboard", dashboard)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)