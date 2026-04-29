from collections import defaultdict, deque
import numpy as np

METERS_PER_PIXEL = 0.058

class StatsTracker:
    def __init__(self, fps=25.0):
        self.fps         = fps
        self.possession  = {0: 0, 1: 0}
        self.dist_px     = defaultdict(float)
        self.team_of     = {}
        self.last_pos    = {}
        self.poss_history = deque(maxlen=int(fps * 5))
        self.speed_hist  = defaultdict(lambda: deque(maxlen=15))

    def update(self, ball_center, players):
        # Possesso
        frame_poss = -1
        if ball_center and players:
            bx, by = ball_center
            best_d, best_team = float('inf'), -1
            for cx, cy, team_id, _ in players:
                d = ((cx-bx)**2 + (cy-by)**2)**0.5
                if d < best_d:
                    best_d, best_team = d, team_id
            if best_d < 150 and best_team in (0, 1):
                self.possession[best_team] += 1
                frame_poss = best_team
        self.poss_history.append(frame_poss)

        # Distanza + velocità
        for cx, cy, team_id, tid in players:
            if tid == -1:
                continue
            if team_id in (0, 1):
                self.team_of[tid] = team_id
            if tid in self.last_pos:
                lx, ly = self.last_pos[tid]
                d = ((cx-lx)**2 + (cy-ly)**2)**0.5
                if d < 80:
                    self.dist_px[tid] += d
                    self.speed_hist[tid].append(d)
            self.last_pos[tid] = (cx, cy)

    def possession_pct(self):
        base = self.possession[0] + self.possession[1]
        if base == 0: return 50.0, 50.0
        return round(self.possession[0]/base*100,1), round(self.possession[1]/base*100,1)

    def recent_possession(self):
        h = list(self.poss_history)
        t0, t1 = h.count(0), h.count(1)
        base = t0 + t1
        if base == 0: return 50.0, 50.0
        return round(t0/base*100,1), round(t1/base*100,1)

    def distance_meters(self):
        d = {0: 0.0, 1: 0.0}
        for tid, px in self.dist_px.items():
            t = self.team_of.get(tid, -1)
            if t in d:
                d[t] += px * METERS_PER_PIXEL
        return d[0], d[1]

    def avg_speed_kmh(self):
        speeds = {0: [], 1: []}
        for tid, hist in self.speed_hist.items():
            t = self.team_of.get(tid, -1)
            if t in speeds and hist:
                kmh = np.mean(hist) * METERS_PER_PIXEL * self.fps * 3.6
                speeds[t].append(kmh)
        s0 = float(np.mean(speeds[0])) if speeds[0] else 0.0
        s1 = float(np.mean(speeds[1])) if speeds[1] else 0.0
        return s0, s1