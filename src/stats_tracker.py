from collections import defaultdict, deque

METERS_PER_PIXEL = 0.06

class StatsTracker:
    def __init__(self):
        self.possession  = {0: 0, 1: 0}
        self.dist_px     = defaultdict(float)
        self.team_of     = {}
        self.last_pos    = {}
        self.poss_history = deque(maxlen=150)  # ultimi 5s a 30fps

    def update(self, ball_center, players):
        frame_poss = -1
        if ball_center is not None and players:
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
            self.last_pos[tid] = (cx, cy)

    def possession_pct(self):
        t0, t1 = self.possession[0], self.possession[1]
        base = t0 + t1
        if base == 0:
            return 50.0, 50.0
        return round(t0/base*100, 1), round(t1/base*100, 1)

    def recent_possession(self):
        """Possesso negli ultimi 5s"""
        h = list(self.poss_history)
        t0 = h.count(0); t1 = h.count(1)
        base = t0 + t1
        if base == 0:
            return 50.0, 50.0
        return round(t0/base*100, 1), round(t1/base*100, 1)

    def distance_meters(self):
        d = {0: 0.0, 1: 0.0}
        for tid, px in self.dist_px.items():
            t = self.team_of.get(tid, -1)
            if t in d:
                d[t] += px * METERS_PER_PIXEL
        return d[0], d[1]

    def player_count_per_team(self, team_of_active):
        d = {0: 0, 1: 0}
        for t in team_of_active.values():
            if t in d:
                d[t] += 1
        return d[0], d[1]