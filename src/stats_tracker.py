from collections import defaultdict

METERS_PER_PIXEL = 0.06  # ~1920px ≈ 105m → 0.055m/px

class StatsTracker:
    def __init__(self):
        self.possession  = {0: 0, 1: 0}   # frame con possesso
        self.dist_px     = defaultdict(float)  # tid → pixel percorsi
        self.team_of     = {}              # tid → team_id
        self.last_pos    = {}              # tid → (cx, cy)

    def update(self, ball_center, players):
        """
        ball_center : (bx, by) oppure None
        players     : lista di (cx, cy, team_id, tid)
        """
        # Possesso
        if ball_center is not None and players:
            bx, by = ball_center
            best_d, best_team = float('inf'), -1
            for cx, cy, tid_team, _ in players:
                d = ((cx-bx)**2 + (cy-by)**2)**0.5
                if d < best_d:
                    best_d, best_team = d, tid_team
            if best_d < 150 and best_team in (0, 1):
                self.possession[best_team] += 1

        # Distanza per ogni giocatore
        for cx, cy, team_id, tid in players:
            if tid == -1:
                continue
            if team_id in (0, 1):
                self.team_of[tid] = team_id
            if tid in self.last_pos:
                lx, ly = self.last_pos[tid]
                d = ((cx-lx)**2 + (cy-ly)**2)**0.5
                if d < 80:          # filtra teleport
                    self.dist_px[tid] += d
            self.last_pos[tid] = (cx, cy)

    def possession_pct(self):
        t0, t1 = self.possession[0], self.possession[1]
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