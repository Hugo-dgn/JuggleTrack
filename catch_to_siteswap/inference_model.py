import numpy as np
import pandas as pd
from math import isclose


def build_tempo_map(events, win=7):
    """
    Estimate a local beat duration T(t) from sorted events (by time).
    Returns t_mid (centers of intervals) and T_local (median-smoothed dt).
    events: list of {"time": float, "ball_id": ...}
    win: window size for rolling median (odd recommended)
    """
    times = np.array([e["time"] for e in events])
    dt = np.diff(times)
    if len(dt) == 0:
        return np.array([]), np.array([])
    # rolling median via pandas
    T_local = pd.Series(dt).rolling(win, center=True, min_periods=1).median().to_numpy()
    t_mid = (times[:-1] + times[1:]) / 2.0
    return t_mid, T_local


def tempo_at_array(ts, t_mid, T_local):
    """Interpolate T at array of times ts."""
    if len(t_mid) == 0:
        return np.ones_like(ts) * (T_local[0] if len(T_local) > 0 else 1.0)
    return np.interp(ts, t_mid, T_local, left=T_local[0], right=T_local[-1])


def integrate_beats(t0, t1, t_mid, T_local, n_samples=30):
    """Approximate integral of dt/T(t) between t0 and t1 by sampling."""
    if t1 <= t0:
        return 0.0
    ts = np.linspace(t0, t1, n_samples)
    Ts = tempo_at_array(ts, t_mid, T_local)
    beats = np.trapz(1.0 / Ts, ts)  # integral dt/T(t)
    return beats


# -------------------------
# Préparer les throws observés (seulement les throws où on connaît le retour)
# -------------------------
def prepare_throws(events):
    """
    From events (sorted by time), build list of throws where next catch for same ball exists.
    Returns:
      throws = list of dicts {"index": idx_in_events, "t0": t0, "t1": t1, "ball_id": id}
    """
    per_ball = {}
    for idx, e in enumerate(events):
        per_ball.setdefault(e["ball_id"], []).append((idx, e["time"]))
    throws = []
    for ball_id, seq in per_ball.items():
        for k in range(len(seq) - 1):
            idx0, t0 = seq[k]
            idx1, t1 = seq[k + 1]
            throws.append({"index": idx0, "t0": t0, "t1": t1, "ball_id": ball_id})
    # sort throws in chronological order of t0
    throws.sort(key=lambda x: x["t0"])
    return throws


def infer_siteswap_global(events, max_period=10, max_s=20, sigma=0.25, win=7):
    """
    Infer most likely repeating siteswap pattern by testing periods L=1..max_period.
    events: list of {"time": float, "ball_id": ...} sorted by time
    Returns best_result dict containing:
      - 'L', 'pattern' (list length L), 'score' (log-likelihood), 'assignments' (s for each throw in chronological order),
        'conf_per_throw' (abs residual / sigma), 'method' ('periodic' or 'fallback')
    If no valid periodic pattern is found, a fallback per-throw rounding is returned.
    """
    # 1) prepare tempo map and throws
    t_mid, T_local = build_tempo_map(events, win=win)
    throws = prepare_throws(events)
    if len(throws) == 0:
        return {
            "L": 0,
            "pattern": [],
            "assignments": [],
            "conf_per_throw": [],
            "score": None,
            "method": "empty",
        }

    # compute observed beats for each throw
    b_obs = []
    for th in throws:
        b = integrate_beats(th["t0"], th["t1"], t_mid, T_local)
        b_obs.append(b)
    b_obs = np.array(b_obs)
    n_throws = len(throws)

    best = None

    for L in range(1, max_period + 1):
        # assign throws to positions modulo L (chronological)
        pos = np.arange(n_throws) % L
        # collect observations per position
        obs_per_pos = {j: [] for j in range(L)}
        for k in range(n_throws):
            obs_per_pos[pos[k]].append(b_obs[k])
        # for each position choose s_j in 0..max_s that maximizes sum log-likelihood
        s_candidates = []
        valid_flag = True
        loglik = 0.0
        for j in range(L):
            obs = np.array(obs_per_pos[j])
            if len(obs) == 0:
                # if no observations for this pos (possible for short sequences), mark invalid
                valid_flag = False
                break
            # compute best integer s in [0..max_s] minimizing sum (obs - s)^2
            # equivalent to maximizing sum exp(- (obs - s)^2 / (2 sigma^2))
            # search over s
            errors = []
            for s in range(0, max_s + 1):
                err = np.sum((obs - s) ** 2)
                errors.append(err)
            errors = np.array(errors)
            best_s = int(np.argmin(errors))
            best_err = float(errors[best_s])
            s_candidates.append(best_s)
            # accumulate log-likelihood (up to const): -0.5 * err / sigma^2
            loglik += -0.5 * best_err / (sigma**2)
        if not valid_flag:
            continue

        # check siteswap constraints: collisions and integer mean
        s_arr = np.array(s_candidates, dtype=int)
        landings = (np.arange(L) + s_arr) % L
        collisions = len(set(landings)) != len(landings)
        mean_balls = s_arr.sum() / float(L)
        mean_ok = float(isclose(round(mean_balls), mean_balls, rel_tol=1e-6))

        if collisions or (not mean_ok):
            # invalid pattern, skip
            continue

        # pattern valid; keep if best score
        if (best is None) or (loglik > best["score"]):
            # compute per-throw assignment (map pos->s)
            assigns = [s_candidates[p] for p in pos]
            # compute per-throw confidence metric (residual / sigma)
            confs = [abs(b_obs[k] - assigns[k]) / sigma for k in range(n_throws)]
            best = {
                "L": L,
                "pattern": s_candidates,
                "score": loglik,
                "assignments": assigns,
                "conf_per_throw": confs,
                "method": "periodic",
            }

    # fallback: if no valid periodic pattern found, do independent rounding (but respecting bounds)
    if best is None:
        assigns = [int(np.clip(round(b), 0, max_s)) for b in b_obs]
        confs = [abs(b_obs[k] - assigns[k]) / sigma for k in range(n_throws)]
        # final validation optional: compute collisions on the chronological sequence assuming L = n_throws
        # but we will just return fallback
        best = {
            "L": n_throws,
            "pattern": None,
            "score": None,
            "assignments": assigns,
            "conf_per_throw": confs,
            "method": "fallback",
        }

    # Finally, return mapping aligned with original events (chronological throws)
    # Also convert assignments to list of ints
    best["assignments"] = list(map(int, best["assignments"]))
    best["conf_per_throw"] = list(best["conf_per_throw"])
    return best


if __name__ == "__main__":
    events = [
        {"time": 0.30, "ball_id": 1},
        {"time": 0.60, "ball_id": 2},
        {"time": 0.90, "ball_id": 3},
        {"time": 1.25, "ball_id": 1},
        {"time": 1.55, "ball_id": 2},
        {"time": 1.90, "ball_id": 3},
        {"time": 2.20, "ball_id": 1},
        {"time": 2.50, "ball_id": 2},
        {"time": 2.80, "ball_id": 3},
    ]
    result = infer_siteswap_global(events, max_period=8, max_s=12, sigma=0.25, win=5)
    print("Method:", result["method"])
    print("Period L:", result["L"])
    print("Pattern (per pos):", result["pattern"])
    print("Assignments (per throw):", result["assignments"])
    print("Conf per throw (residual / sigma):", np.round(result["conf_per_throw"], 2))
