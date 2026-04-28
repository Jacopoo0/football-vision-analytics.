import cv2
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
INPUT_VID  = BASE_DIR / "data" / "raw" / "full_match.mp4"
OUTPUT_VID = BASE_DIR / "data" / "raw" / "input_vid.mp4"

START_MIN  = 16   # minuto di inizio
END_MIN    = 17   # minuto di fine

# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(str(INPUT_VID))
if not cap.isOpened():
    print(f"Impossibile aprire: {INPUT_VID}")
    exit()

fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_frame = int(START_MIN * 60 * fps)
end_frame   = int(END_MIN   * 60 * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

out = cv2.VideoWriter(
    str(OUTPUT_VID),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

print(f"Taglio: {START_MIN}:00 → {END_MIN}:00  ({end_frame - start_frame} frame)")

while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
    ok, frame = cap.read()
    if not ok:
        break
    out.write(frame)

cap.release()
out.release()
print(f"Salvato: {OUTPUT_VID}")