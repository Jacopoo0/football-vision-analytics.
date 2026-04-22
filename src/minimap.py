import cv2
import numpy as np


MINIMAP_WIDTH = 800
MINIMAP_HEIGHT = 520
PADDING = 40

FIELD_COLOR = (34, 85, 34)
LINE_COLOR = (255, 255, 255)
PENALTY_COLOR = (255, 255, 255)


def create_minimap():
    minimap = np.zeros((MINIMAP_HEIGHT, MINIMAP_WIDTH, 3), dtype=np.uint8)
    minimap[:] = FIELD_COLOR

    # Bordo campo
    cv2.rectangle(minimap, (PADDING, PADDING),
                  (MINIMAP_WIDTH - PADDING, MINIMAP_HEIGHT - PADDING),
                  LINE_COLOR, 2)

    # Linea centrocampo
    cv2.line(minimap,
             (MINIMAP_WIDTH // 2, PADDING),
             (MINIMAP_WIDTH // 2, MINIMAP_HEIGHT - PADDING),
             LINE_COLOR, 2)

    # Cerchio centrocampo
    cv2.circle(minimap,
               (MINIMAP_WIDTH // 2, MINIMAP_HEIGHT // 2),
               55, LINE_COLOR, 2)

    # Punto centrocampo
    cv2.circle(minimap,
               (MINIMAP_WIDTH // 2, MINIMAP_HEIGHT // 2),
               4, LINE_COLOR, -1)

    field_w = MINIMAP_WIDTH - 2 * PADDING
    field_h = MINIMAP_HEIGHT - 2 * PADDING

    # Area di rigore sinistra
    pen_w = int(field_w * 0.12)
    pen_h = int(field_h * 0.45)
    pen_y1 = MINIMAP_HEIGHT // 2 - pen_h // 2
    pen_y2 = MINIMAP_HEIGHT // 2 + pen_h // 2
    cv2.rectangle(minimap,
                  (PADDING, pen_y1),
                  (PADDING + pen_w, pen_y2),
                  PENALTY_COLOR, 2)

    # Area piccola sinistra
    small_w = int(field_w * 0.05)
    small_h = int(field_h * 0.22)
    small_y1 = MINIMAP_HEIGHT // 2 - small_h // 2
    small_y2 = MINIMAP_HEIGHT // 2 + small_h // 2
    cv2.rectangle(minimap,
                  (PADDING, small_y1),
                  (PADDING + small_w, small_y2),
                  PENALTY_COLOR, 2)

    # Area di rigore destra
    cv2.rectangle(minimap,
                  (MINIMAP_WIDTH - PADDING - pen_w, pen_y1),
                  (MINIMAP_WIDTH - PADDING, pen_y2),
                  PENALTY_COLOR, 2)

    # Area piccola destra
    cv2.rectangle(minimap,
                  (MINIMAP_WIDTH - PADDING - small_w, small_y1),
                  (MINIMAP_WIDTH - PADDING, small_y2),
                  PENALTY_COLOR, 2)

    # Dischetti rigore
    pen_spot_offset = int(field_w * 0.08)
    cv2.circle(minimap,
               (PADDING + pen_spot_offset, MINIMAP_HEIGHT // 2),
               4, LINE_COLOR, -1)
    cv2.circle(minimap,
               (MINIMAP_WIDTH - PADDING - pen_spot_offset, MINIMAP_HEIGHT // 2),
               4, LINE_COLOR, -1)

    return minimap


def draw_player_on_minimap(minimap, map_x, map_y, track_id, color=(0, 255, 0)):
    cv2.circle(minimap, (map_x, map_y), 7, (0, 0, 0), -1)
    cv2.circle(minimap, (map_x, map_y), 5, color, -1)
    cv2.putText(
        minimap,
        str(track_id),
        (map_x + 8, map_y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1
    )