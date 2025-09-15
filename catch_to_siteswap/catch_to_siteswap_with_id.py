import numpy as np
import pandas as pd


def build_tempo_map(events, win=5):
    times = np.array([e["time"] for e in events])
    dt = np.diff(times)
    T_local = pd.Series(dt).rolling(win, center=True, min_periods=1).median().to_numpy()
    t_mid = (times[:-1] + times[1:]) / 2
    return t_mid, T_local


def tempo_at(t, t_mid, T_local):
    return np.interp(t, t_mid, T_local, left=T_local[0], right=T_local[-1])


def compute_siteswap(events, tol=0.25):
    t_mid, T_local = build_tempo_map(events)

    per_ball = {}
    for idx, e in enumerate(events):
        per_ball.setdefault(e["ball_id"], []).append((idx, e["time"]))

    siteswap = []
    conf = []
    indices = []  # pour garder la correspondance avec les événements

    for ball_id, seq in per_ball.items():
        for k in range(len(seq) - 1):  # on s'arrête avant le dernier catch
            i, t0 = seq[k]
            _, t1 = seq[k + 1]
            dt = t1 - t0
            ts = np.linspace(t0, t1, 20)
            Ts = tempo_at(ts, t_mid, T_local)
            mean_T = Ts.mean()
            beats = dt / mean_T
            s = int(np.round(beats))
            is_confident = abs(beats - s) <= tol
            siteswap.append(s)
            conf.append(is_confident)
            indices.append(i)

    # Réordonner dans l'ordre chronologique des throws
    order = np.argsort(indices)
    siteswap = [siteswap[o] for o in order]
    conf = [conf[o] for o in order]

    return siteswap, conf


def validate_siteswap(siteswap):
    """
    Vérifie la cohérence d'une séquence de siteswap.
    siteswap: liste d'entiers
    Retourne un dict avec diagnostic
    """
    n = len(siteswap)
    total = sum(siteswap)

    # condition 1 : moyenne entière
    mean_balls = total / n
    balls_ok = mean_balls.is_integer()

    # condition 2 : pas de collisions
    landings = [(i + siteswap[i]) % n for i in range(n)]
    collisions = len(landings) != len(set(landings))

    # rapport
    result = {
        "length": n,
        "sequence": siteswap,
        "mean_balls": mean_balls,
        "balls_ok": balls_ok,
        "collisions": collisions,
        "valid": balls_ok and not collisions,
    }
    return result


# Exemple
# events = [
#    {"time": 0.30, "ball_id": 1},
#    {"time": 0.60, "ball_id": 2},
#    {"time": 0.90, "ball_id": 3},
#    {"time": 1.25, "ball_id": 1},
#    {"time": 1.55, "ball_id": 2},
#    {"time": 1.90, "ball_id": 3},
#    {"time": 2.20, "ball_id": 1},
# ]
#
# s, conf = compute_siteswap(events)
# report = validate_siteswap(s)
#
# if report["valid"]:
#    print("Siteswap cohérent :", s, "(", report["mean_balls"], "balles )")
# else:
#    print("⚠️ Problème détecté :", report)
