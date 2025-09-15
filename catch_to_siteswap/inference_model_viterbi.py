import math
import numpy as np
import pandas as pd


def build_tempo_map(events, win=7):
    """
    events: list of dict {"time": float, "ball_id": ...} sorted by time
    win: window for rolling median (odd recommended)
    returns: t_mid (centers of intervals), T_local (median-smoothed dt)
    """
    times = np.array([e["time"] for e in events])
    dt = np.diff(times)
    if len(dt) == 0:
        return np.array([]), np.array([])
    T_local = pd.Series(dt).rolling(win, center=True, min_periods=1).median().to_numpy()
    t_mid = (times[:-1] + times[1:]) / 2.0
    return t_mid, T_local


def tempo_at_array(ts, t_mid, T_local):
    if len(t_mid) == 0:
        # fallback: if no dt available, assume 1.0s beat (unlikely)
        return np.ones_like(ts) * (T_local[0] if len(T_local) > 0 else 1.0)
    return np.interp(ts, t_mid, T_local, left=T_local[0], right=T_local[-1])


def integrate_beats(t0, t1, t_mid, T_local, n_samples=30):
    """Approximate integral of dt/T(t) between t0 and t1 by sampling."""
    if t1 <= t0:
        return 0.0
    ts = np.linspace(t0, t1, n_samples)
    Ts = tempo_at_array(ts, t_mid, T_local)
    beats = np.trapz(1.0 / Ts, ts)  # integral dt / T(t)
    return beats


def prepare_throws(events):
    """
    From events build list of throws where next catch for same ball exists.
    Returns throws: list of dicts {'event_index','t0','t1','ball_id'}
    """
    per_ball = {}
    for idx, e in enumerate(events):
        per_ball.setdefault(e["ball_id"], []).append((idx, e["time"]))
    throws = []
    for ball_id, seq in per_ball.items():
        for k in range(len(seq) - 1):
            idx0, t0 = seq[k]
            idx1, t1 = seq[k + 1]
            throws.append({"event_index": idx0, "t0": t0, "t1": t1, "ball_id": ball_id})
    # sort throws chronologically
    throws.sort(key=lambda x: x["t0"])
    return throws


def viterbi_siteswap_beam(
    events, max_s=12, sigma=0.25, sigma_per_throw=None, win=7, beam_width=200, W=24
):
    """
    Beam-search Viterbi-like DP to assign integer siteswap heights to throws.
    - events: list of {"time":float, "ball_id":...} sorted by time
    - max_s: max siteswap height to consider
    - sigma: default observation std dev (in beats) if sigma_per_throw not provided
    - sigma_per_throw: optional list/array of length n_throws with per-throw std dev
    - win: window for tempo-map smoothing
    - beam_width: number of candidate states kept at each step
    - W: occupancy window size (must be >= max_s). If throws have s>W increase W.
    Returns dict with:
      - "assignments": list of ints (s for each throw, chronological)
      - "residuals": observed_beats - assigned_s
      - "score": log-likelihood
      - "throws": metadata list (t0,t1,ball_id,event_index)
      - "method": "beam_viterbi" or "fallback"/"empty"
    """
    # tempo map and throws
    t_mid, T_local = build_tempo_map(events, win=win)
    throws = prepare_throws(events)
    if len(throws) == 0:
        return {
            "assignments": [],
            "residuals": [],
            "score": 0.0,
            "method": "empty",
            "throws": [],
        }

    # compute observed beats per throw and absolute cumulative beat (for base)
    # we take baseline at first throw time
    t0_start = throws[0]["t0"]
    b_obs = []
    B_abs = []
    for th in throws:
        b = integrate_beats(th["t0"], th["t1"], t_mid, T_local)
        b_obs.append(b)
        B_abs.append(integrate_beats(t0_start, th["t0"], t_mid, T_local))
    b_obs = np.array(b_obs)
    B_abs = np.array(B_abs)
    base = np.floor(B_abs + 1e-9).astype(int)  # integer base beat index

    n = len(throws)
    if sigma_per_throw is not None:
        sigma_arr = np.array(sigma_per_throw, dtype=float)
        if len(sigma_arr) != n:
            raise ValueError("sigma_per_throw must match number of throws")
        # avoid zeros
        sigma_arr = np.maximum(sigma_arr, 1e-6)
    else:
        sigma_arr = np.full(n, float(sigma))

    if W < max_s:
        W = max_s + 2

    # DP states: dict mask -> (score, path)
    # mask is occupancy bitmask for offsets 0..W-1 relative to current base
    states = {0: (0.0, [])}  # start: empty occupancy, zero score

    for k in range(n):
        base_k = base[k]
        next_tmp = {}  # (new_mask) -> (score, path)
        for mask, (score, path) in states.items():
            # try all candidate s
            for s in range(0, max_s + 1):
                if s >= W:
                    break
                # if landing offset s already occupied -> collision
                if ((mask >> s) & 1) != 0:
                    continue
                residual = b_obs[k] - s
                inc = -0.5 * (residual**2) / (sigma_arr[k] ** 2)
                new_score = score + inc
                new_mask = mask | (1 << s)
                prev = next_tmp.get(new_mask)
                if (prev is None) or (new_score > prev[0]):
                    next_tmp[new_mask] = (new_score, path + [s])
        # if last throw, choose best among next_tmp
        if k == n - 1:
            if not next_tmp:
                # no feasible state (shouldn't happen usually) -> fallback
                assigns = [int(np.clip(round(b), 0, max_s)) for b in b_obs]
                residuals = list(b_obs - np.array(assigns))
                score = sum(
                    [
                        -0.5 * (r**2) / (sigma_arr[i] ** 2)
                        for i, r in enumerate(residuals)
                    ]
                )
                return {
                    "assignments": assigns,
                    "residuals": residuals,
                    "score": score,
                    "method": "fallback",
                    "throws": throws,
                }
            # pick best by score
            best_mask, (best_score, best_path) = max(
                next_tmp.items(), key=lambda kv: kv[1][0]
            )
            residuals = [b_obs[i] - best_path[i] for i in range(len(best_path))]
            return {
                "assignments": best_path,
                "residuals": residuals,
                "score": best_score,
                "method": "beam_viterbi",
                "throws": throws,
            }
        # else shift masks to next base (base[k+1]) and prune to beam_width
        shift = base[k + 1] - base_k
        new_states = {}
        for mask, (score, path) in next_tmp.items():
            if shift <= 0:
                shifted = mask  # no forward shift (rare if tempo map wobbles)
            else:
                if shift >= W:
                    shifted = 0
                else:
                    shifted = mask >> shift
            prev = new_states.get(shifted)
            if (prev is None) or (score > prev[0]):
                new_states[shifted] = (score, path)
        # prune to beam_width best states
        sorted_states = sorted(new_states.items(), key=lambda x: -x[1][0])[:beam_width]
        states = {mask: val for mask, val in sorted_states}

    # fallback - should not reach here
    assigns = [int(np.clip(round(b), 0, max_s)) for b in b_obs]
    residuals = list(b_obs - np.array(assigns))
    score = sum([-0.5 * (r**2) / (sigma_arr[i] ** 2) for i, r in enumerate(residuals)])
    return {
        "assignments": assigns,
        "residuals": residuals,
        "score": score,
        "method": "fallback",
        "throws": throws,
    }


def validate_siteswap_from_assignments(assignments):
    """
    Given chronological list of assignments s_i (one per throw observed),
    test simple siteswap constraints by checking length n, and whether
    a periodic pattern exists with no collisions for period L = len(assignments).
    This is a lightweight check (more advanced checks could search for minimal period L).
    """
    n = len(assignments)
    if n == 0:
        return {"valid": False, "reason": "empty"}
    L = n
    s = list(assignments)
    landings = [(i + s[i]) % L for i in range(L)]
    collisions = len(landings) != len(set(landings))
    mean_balls = sum(s) / float(L)
    mean_ok = float(round(mean_balls) == mean_balls)
    return {
        "length": L,
        "mean_balls": mean_balls,
        "mean_ok": mean_ok,
        "collisions": collisions,
        "valid": (not collisions and mean_ok),
    }


if __name__ == "__main__":
    T0 = 0.33
    times = []
    ball_ids = []
    t = 0.0
    for n in range(9):
        T = T0 * (1.0 + 0.05 * math.sin(n / 3.0))  # slow wobble
        times.append(t)
        ball_ids.append((n % 3) + 1)
        t += T
    events = [{"time": times[i], "ball_id": ball_ids[i]} for i in range(len(times))]
    res = viterbi_siteswap_beam(events, max_s=6, sigma=0.2, win=5, beam_width=300, W=12)
    print("Method:", res["method"])
    print("Assignments (per throw):", res["assignments"])
    print("Residuals:", np.round(res["residuals"], 3))
    print("Score:", res["score"])
    print("Validation:", validate_siteswap_from_assignments(res["assignments"]))
