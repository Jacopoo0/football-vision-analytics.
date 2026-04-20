import cv2
import json
import numpy as np
from collections import defaultdict, deque


class TeamColorClassifier:
    def __init__(self, history_size=10, other_absolute_threshold=40.0, ambiguity_margin=8.0):
        self.team_prototypes = {}
        self.track_history = defaultdict(lambda: deque(maxlen=history_size))
        self.stable_team_by_track = {}
        self.history_size = history_size
        self.other_absolute_threshold = other_absolute_threshold
        self.ambiguity_margin = ambiguity_margin

    def extract_jersey_color(self, frame, box):
        x1, y1, x2, y2 = map(int, box)

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return None

        roi_x1 = x1 + int(w * 0.25)
        roi_x2 = x1 + int(w * 0.75)
        roi_y1 = y1 + int(h * 0.18)
        roi_y2 = y1 + int(h * 0.48)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        green_mask = cv2.inRange(hsv, (35, 25, 25), (95, 255, 255))
        non_green_mask = cv2.bitwise_not(green_mask)

        pixels = lab[non_green_mask > 0]

        if len(pixels) < 20:
            pixels = lab.reshape(-1, 3)

        mean_color = np.mean(pixels, axis=0)
        return mean_color

    def set_team_prototypes(self, team0_color, team1_color):
        self.team_prototypes = {
            0: np.array(team0_color, dtype=np.float32),
            1: np.array(team1_color, dtype=np.float32)
        }

    def save_prototypes(self, output_path):
        data = {
            "team_0": self.team_prototypes[0].tolist(),
            "team_1": self.team_prototypes[1].tolist()
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_prototypes(self, input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.team_prototypes = {
            0: np.array(data["team_0"], dtype=np.float32),
            1: np.array(data["team_1"], dtype=np.float32)
        }

    def _raw_predict(self, color):
        if color is None or len(self.team_prototypes) != 2:
            return None

        color = np.array(color, dtype=np.float32)

        dist_0 = float(np.linalg.norm(color - self.team_prototypes[0]))
        dist_1 = float(np.linalg.norm(color - self.team_prototypes[1]))

        best_team = 0 if dist_0 <= dist_1 else 1
        best_dist = min(dist_0, dist_1)
        second_dist = max(dist_0, dist_1)

        if best_dist > self.other_absolute_threshold:
            return 2

        if (second_dist - best_dist) < self.ambiguity_margin:
            return 2

        return best_team

    def _smooth_prediction(self, track_id):
        history = list(self.track_history[track_id])

        if not history:
            return None

        count_0 = history.count(0)
        count_1 = history.count(1)
        count_2 = history.count(2)

        previous_stable = self.stable_team_by_track.get(track_id)

        if previous_stable in [0, 1]:
            if history[-1] == previous_stable:
                return previous_stable

            if history.count(previous_stable) >= max(3, len(history) // 3):
                return previous_stable

            if len(history) >= 5 and history[-5:].count(2) >= 4:
                return 2

            other_team = 1 if previous_stable == 0 else 0
            other_votes = count_1 if other_team == 1 else count_0
            prev_votes = count_0 if previous_stable == 0 else count_1

            if other_votes >= prev_votes + 3 and other_votes >= 5:
                self.stable_team_by_track[track_id] = other_team
                return other_team

            return previous_stable

        best_team = 0 if count_0 >= count_1 else 1
        best_votes = max(count_0, count_1)

        if best_votes >= 4:
            self.stable_team_by_track[track_id] = best_team
            return best_team

        if count_2 >= 5:
            return 2

        return best_team

    def predict_team(self, track_id, color):
        raw_label = self._raw_predict(color)

        if raw_label is None:
            return None

        self.track_history[track_id].append(raw_label)
        return self._smooth_prediction(track_id)

    def get_team_color(self, team_id):
        if team_id == 0:
            return (255, 0, 0)
        if team_id == 1:
            return (0, 0, 255)
        if team_id == 2:
            return (0, 255, 255)
        return (0, 255, 0)

    def get_team_label(self, team_id):
        if team_id == 0:
            return "TEAM 0"
        if team_id == 1:
            return "TEAM 1"
        if team_id == 2:
            return "OTHERS"
        return "UNK"