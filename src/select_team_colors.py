import cv2
import json
import numpy as np
from pathlib import Path
from team_classifier import _dominant_lab

BASE_DIR     = Path(__file__).resolve().parent.parent
VIDEO_PATH   = BASE_DIR / "data" / "raw" / "input_vid.mp4"
OUTPUT_JSON  = BASE_DIR / "team_colors.json"
SAMPLES_EACH = 12
START_FRAME  = 150

COLOR_UI = {0: (100, 80, 220), 1: (60, 180, 60)}
NAMES    = {0: "TEAM 0", 1: "TEAM 1"}

samples      = {0: [], 1: []}
current_team = 0

cap = cv2.VideoCapture(str(VIDEO_PATH))
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
ok, frame = cap.read()
cap.release()

if not ok:
    print("Impossibile leggere il video.")
    exit()

vis     = frame.copy()
drawing = False
ix, iy  = -1, -1


def draw_ui(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 44), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    t0  = len(samples[0])
    t1  = len(samples[1])
    msg = (f"T0:{t0}/{SAMPLES_EACH}  T1:{t1}/{SAMPLES_EACH}  "
           f"| Selezione: {NAMES[current_team]}  "
           f"| TAB=cambia squadra  INVIO=salva  ESC=esci")
    cv2.putText(img, msg, (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, COLOR_UI[current_team], 1, cv2.LINE_AA)


def on_mouse(event, x, y, flags, param):
    global drawing, ix, iy, vis
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy  = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        vis = frame.copy()
        cv2.rectangle(vis, (ix, iy), (x, y), COLOR_UI[current_team], 2)
        draw_ui(vis)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx, ry  = min(ix, x), min(iy, y)
        rw, rh  = abs(x - ix), abs(y - iy)
        if rw > 10 and rh > 15:
            color = _dominant_lab(frame, (rx, ry, rx + rw, ry + rh))
            if color is not None:
                samples[current_team].append(color.tolist())
                print(f"  Team {current_team}: {len(samples[current_team])}/{SAMPLES_EACH}")
        vis = frame.copy()
        draw_ui(vis)


cv2.namedWindow("Select Team Colors")
cv2.setMouseCallback("Select Team Colors", on_mouse)
draw_ui(vis)

print("Disegna un rettangolo sulla MAGLIA.")
print("TAB=cambia squadra | INVIO=salva | ESC=esci")

while True:
    cv2.imshow("Select Team Colors", vis)
    key = cv2.waitKey(20) & 0xFF
    if key == 9:   # TAB
        current_team = 1 - current_team
        vis = frame.copy()
        draw_ui(vis)
        print(f">>> {NAMES[current_team]}")
    elif key == 13:  # INVIO
        if len(samples[0]) < 3 or len(samples[1]) < 3:
            print(f"Troppo pochi campioni: T0={len(samples[0])} T1={len(samples[1])}")
            continue
        data = {f"team_{k}": v for k, v in samples.items()}
        with open(OUTPUT_JSON, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Salvato: {OUTPUT_JSON}")
        break
    elif key == 27:  # ESC
        print("Annullato.")
        break

cv2.destroyAllWindows()