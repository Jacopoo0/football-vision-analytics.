import cv2

video_path = "data/raw/input_vid.mp4"
timestamps = [3.0, 14.0, 24.0]

current_index = 0
selected_points = []
frame_original = None
cap = None


def load_frame_at_time(sec):
    global cap
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    success, frame = cap.read()
    return success, frame


def redraw():
    frame_copy = frame_original.copy()

    cv2.putText(
        frame_copy,
        f"Segmento {current_index + 1} - tempo {timestamps[current_index]}s",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_copy,
        "Ordine: alto-sinistra, alto-destra, basso-sinistra, basso-destra",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame_copy,
        "Tasti: r=reset  n=next  q=quit",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    for i, (x, y) in enumerate(selected_points):
        cv2.circle(frame_copy, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(
            frame_copy,
            str(i + 1),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("Select Points", frame_copy)


def mouse_callback(event, x, y, flags, param):
    global selected_points

    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append((x, y))
        redraw()


cap = cv2.VideoCapture(video_path)

success, frame_original = load_frame_at_time(timestamps[current_index])

if not success:
    print("Errore nel leggere il video")
    cap.release()
    exit()

cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", mouse_callback)
redraw()

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        selected_points = []
        redraw()

    elif key == ord("n"):
        if len(selected_points) != 4:
            print("Devi selezionare esattamente 4 punti prima di andare avanti.")
            continue

        print(f"SEGMENTO {current_index + 1} ({timestamps[current_index]}s): {selected_points}")

        current_index += 1
        if current_index >= len(timestamps):
            print("Fatto. Copia i 3 blocchi di punti dentro homography.py")
            break

        selected_points = []
        success, frame_original = load_frame_at_time(timestamps[current_index])

        if not success:
            print("Errore nel leggere il frame successivo")
            break

        redraw()

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)