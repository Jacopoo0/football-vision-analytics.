import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
VIDEO_PATH  = BASE_DIR / "data" / "raw" / "input_vid.mp4"
MODEL_KP_PT = BASE_DIR / "models" / "soccana_keypoint.pt"

model = YOLO(str(MODEL_KP_PT))
cap   = cv2.VideoCapture(str(VIDEO_PATH))

# Vai al frame 60 (più stabile del frame 1)
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
ok, frame = cap.read()
cap.release()

if not ok:
    print("Frame non trovato")
    exit()

results = model.predict(frame, imgsz=640, conf=0.3, device="cuda", verbose=False)[0]

if results.keypoints is None or len(results.keypoints.data) == 0:
    print("Nessun keypoint trovato")
    exit()

kps = results.keypoints.data[0].cpu().numpy()  # (N, 3)
print(f"Numero keypoint: {len(kps)}")

# Disegna ogni keypoint con il suo indice
vis = frame.copy()
for i, (x, y, conf) in enumerate(kps):
    if conf > 0.3 and x > 0 and y > 0:
        px, py = int(x), int(y)
        cv2.circle(vis, (px, py), 5, (0, 255, 255), -1)
        cv2.putText(vis, str(i), (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        print(f"  KP {i:2d}: pixel=({px:4d},{py:4d})  conf={conf:.2f}")

cv2.imwrite(str(BASE_DIR / "debug_keypoints.jpg"), vis)
print(f"\nSalvato: {BASE_DIR / 'debug_keypoints.jpg'}")