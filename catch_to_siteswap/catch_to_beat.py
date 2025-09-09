import numpy as np


def estimate_beat(times, refine=True):
    # times: ascending list/array of timestamps (s)
    dt = np.diff(times)
    if len(dt) == 0:
        return None
    # use median of the lower quantile as initial beat (robust to occasional long throws)
    q = np.quantile(dt, 0.15)  # 15%-quantile
    T0 = max(q, np.median(dt) * 0.5)  # fallback
    if not refine:
        return T0
    # refine by local search around T0
    bestT = T0
    best_score = 1e9
    factors = np.linspace(0.8, 1.25, 91)  # search range
    for f in factors:
        T = T0 * f
        b = dt / T
        # score: sum squared distance to nearest integer for each interval
        score = np.sum((b - np.round(b)) ** 2)
        if score < best_score:
            best_score = score
            bestT = T
    return bestT


def timestamps_to_siteswap(times, tol_beats=0.25, max_s=20):
    """
    times: list of timestamps (seconds), length N
    tol_beats: allowable fractional distance to nearest integer (in beats)
    returns: list of s_i (length N-1), conf list of booleans
    """
    times = np.array(times)
    if len(times) < 2:
        return [], []
    T = estimate_beat(times)
    dt = np.diff(times)
    b = dt / T
    s = np.round(b).astype(int)
    conf = np.abs(b - s) <= tol_beats
    # clamp s
    s = np.clip(s, 0, max_s)
    return list(s), list(conf), T
