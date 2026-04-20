import cv2
from ultralytics import YOLO
from team_classifier import TeamColorClassifier

video_path = "data/raw/input_vid.mp4"
output_path = "team_colors.json"

model = YOLO("yolov8n.pt")
classifier = TeamColorClassifier()

cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if not success:
    raise ValueError("Impossibile leggere il primo frame del video.")

result = model(frame, conf=0.35, imgsz=640)[0]

boxes = []
if result.boxes is not None and result.boxes.cls is not None:
    all_boxes = result.boxes.xyxy.cpu().tolist()
    all_cls = result.boxes.cls.int().cpu().tolist()

    for box, cls_id in zip(all_boxes, all_cls):
        if cls_id != 0:
            continue

        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        boxes.append(box)

selected = {}
current_team_to_select = 0


def find_box_from_click(x, y):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i, box
    return None, None


def redraw(highlight_idx=None):
    canvas = frame.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        color = (0, 255, 0)
        thickness = 2

        if i == highlight_idx:
            color = (255, 255, 255)
            thickness = 3

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            canvas,
            f"P{i}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    if 0 not in selected:
        msg = "Clicca un giocatore della TEAM 0"
        msg_color = (255, 0, 0)
    elif 1 not in selected:
        msg = "Clicca un giocatore della TEAM 1"
        msg_color = (0, 0, 255)
    else:
        msg = "Premi S per salvare, R per rifare, Q per uscire"
        msg_color = (0, 255, 255)

    cv2.putText(
        canvas,
        msg,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        msg_color,
        2
    )

    if 0 in selected:
        cv2.putText(
            canvas,
            f"TEAM 0 -> P{selected[0]['box_idx']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    if 1 in selected:
        cv2.putText(
            canvas,
            f"TEAM 1 -> P{selected[1]['box_idx']}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Select Team Colors", canvas)


def mouse_callback(event, x, y, flags, param):
    global current_team_to_select

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    box_idx, box = find_box_from_click(x, y)

    if box is None:
        return

    jersey_color = classifier.extract_jersey_color(frame, box)
    if jersey_color is None:
        return

    selected[current_team_to_select] = {
        "box_idx": box_idx,
        "box": box,
        "color": jersey_color
    }

    if current_team_to_select == 0:
        current_team_to_select = 1

    redraw(highlight_idx=box_idx)


cv2.namedWindow("Select Team Colors")
cv2.setMouseCallback("Select Team Colors", mouse_callback)
redraw()

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        selected = {}
        current_team_to_select = 0
        redraw()

    elif key == ord("s"):
        if 0 not in selected or 1 not in selected:
            print("Devi selezionare entrambe le squadre prima di salvare.")
            continue

        classifier.set_team_prototypes(
            selected[0]["color"],
            selected[1]["color"]
        )
        classifier.save_prototypes(output_path)

        print(f"Prototipi salvati in {output_path}")
        print("TEAM 0:", selected[0]["color"])
        print("TEAM 1:", selected[1]["color"])
        break

cv2.destroyAllWindows()
cv2.waitKey(1)