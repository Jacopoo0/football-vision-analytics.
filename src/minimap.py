import cv2
import numpy as np

MINIMAP_WIDTH = 500
MINIMAP_HEIGHT = 320
PADDING = 20


def create_minimap():
    minimap = np.zeros((MINIMAP_HEIGHT, MINIMAP_WIDTH, 3), dtype=np.uint8)
    minimap[:] = (40, 120, 40)

    cv2.rectangle(
        minimap,
        (PADDING, PADDING),
        (MINIMAP_WIDTH - PADDING, MINIMAP_HEIGHT - PADDING),
        (255, 255, 255),
        2
    )

    cv2.line(
        minimap,
        (MINIMAP_WIDTH // 2, PADDING),
        (MINIMAP_WIDTH // 2, MINIMAP_HEIGHT - PADDING),
        (255, 255, 255),
        2
    )

    cv2.circle(
        minimap,
        (MINIMAP_WIDTH // 2, MINIMAP_HEIGHT // 2),
        40,
        (255, 255, 255),
        2
    )

    return minimap


def draw_player_on_minimap(minimap, map_x, map_y, track_id, color=(0, 255, 0)):
    cv2.circle(minimap, (map_x, map_y), 6, color, -1)
    cv2.putText(
        minimap,
        str(track_id),
        (map_x + 7, map_y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )