import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import pandas as pd
import trackpy as tp
from filterpy.kalman import IMMEstimator, KalmanFilter
from tqdm.auto import tqdm

import utils

def distance_loss(points, trajectories):
    new_pos = [(x, y) for x, y, _ in points]
    old_circles = [t[-1] for t in trajectories]
    old_pos = [(x, y) for x, y, _ in old_circles]
    new_pos = np.array(new_pos)
    old_pos = np.array(old_pos)
    distances = cdist(new_pos, old_pos, metric='euclidean')
    return distances

def poly_loss(points, tarjectories, order):
    window = 30
    predictions = []
    next_t = points[0][2]
    for t in tarjectories:
        x = np.array([x for x, y, t in t])[-window:]
        y = np.array([y for x, y, t in t])[-window:]
        t = np.array([t for x, y, t in t])[-window:]
        
        if len(t) > 1:
            delta_t = next_t - t[-1]
            ts = t - np.min(t)
            
            coeffs_x = np.polyfit(ts, x, order)
            coeffs_y = np.polyfit(ts, y, order)

            t_next = ts[-1] + delta_t
            x_next = np.polyval(coeffs_x, t_next)
            y_next = np.polyval(coeffs_y, t_next)
            
            predictions.append((x_next, y_next))
        else:
            predictions.append((x[-1], y[-1]))
    
    predictions = np.array(predictions)
    new_pos = [(x, y) for x, y, _ in points]
    distances = cdist(new_pos, predictions, metric='euclidean')
    return distances

def track(all_circles):
    trajectories = []
    for idx, circles in enumerate(all_circles):
        points = [(x, y, idx) for x, y, _ in circles]
        if len(trajectories) == 0:
            for point in points:
                trajectories.append([point])
        else:
            if len(points) == 0:
                continue
            distances = distance_loss(points, trajectories)
            predictions = poly_loss(points, trajectories, 1)
            row_ind, col_ind = linear_sum_assignment(predictions)

            for i, j in zip(row_ind, col_ind):
                pred = predictions[i, j]
                dist = distances[i, j]
                if pred > 2*dist:
                    trajectories.append([point])
                else:
                    point = points[i]
                    trajectories[j].append(point)
            
            for i, point in enumerate(points):
                if i not in row_ind:
                    trajectories.append([point])
    
    fill_trajectories = []
    n = len(all_circles)
    for t in trajectories:
        fill_trajectory = []
        i = 0
        for j in range(n):
            if i >= len(t):
                fill_trajectory.append((None, None))
            else:
                x, y, idx = t[i]
                if idx == j:
                    fill_trajectory.append((x, y))
                    i += 1
                else:
                    fill_trajectory.append((None, None))
        
        fill_trajectories.append(fill_trajectory)
        
    return np.array(fill_trajectories)

#########

def brownian(detections):
    trajectories = tp.link_df(detections, search_range=100, memory=100, adaptive_step=0.95,
           adaptive_stop=2)
    
    return trajectories

####


def make_kf_free_fall(dt, g, meas_var=2.0, process_var=1e-4):
    kf = KalmanFilter(dim_x=7, dim_z=3)
    kf.F = np.array([
        [1, 0, dt, 0, 0, 0, 0],
        [0, 1, 0, dt, 0, 0,  0.5*dt**2],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, dt],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    kf.H = np.array([[1,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0],
                     [0,0,0,0,0,0,1]])
    kf.R = np.eye(3) * meas_var
    kf.Q = np.eye(7) * process_var
    kf.P = np.eye(7) * 100.0
    return kf

def make_kf_in_hand(dt, meas_var=2.0, process_var=1e-2):
    kf = KalmanFilter(dim_x=7, dim_z=3)
    kf.F = np.array([
        [1, 0, dt, 0, 0.5*dt**2, 0, 0],
        [0, 1, 0, dt, 0, 0.5*dt**2, 0],
        [0, 0, 1, 0, dt, 0, 0],
        [0, 0, 0, 1, 0, dt, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])
    kf.H = np.array([[1,0,0,0,0,0,0],
                     [0,1,0,0,0,0,0],
                     [0,0,0,0,0,0,1]])
    kf.R = np.eye(3) * meas_var
    kf.Q = np.eye(7) * process_var
    kf.P = np.eye(7) * 100.0
    return kf

def make_imm(dt, g, meas_var=10, process_var=2):
    kf_ca = make_kf_free_fall(dt, g, meas_var, process_var=process_var)
    kf_cv = make_kf_in_hand(dt, meas_var, process_var=process_var)
    mu = np.array([0.1, 0.9])
    trans = np.array([[0.95, 0.05],  # free-fall → mostly stays
                      [0.5, 0.5]])   # in-hand → 50% chance to switch out
    return IMMEstimator([kf_ca, kf_cv], mu, trans)

# --- Tracking function ---
class Track:
    def __init__(self, imm, track_id):
        self.imm = imm
        self.id = track_id
        self.age = 0   # frames since last detection

def kalman(df, n_balls, dt=1, meas_var=10.0, process_var=2, max_distance=1000000):
    """
    Multi-object IMM tracker with fixed maximum number of juggling balls.
    - Tracks are never deleted (no max_age).
    - New tracks are initialized only for unmatched detections,
      up to n_balls total.
    
    df: DataFrame with ['frame','x','y','r']
    Returns: DataFrame with ['frame','x','y','r','particle']
    """

    frames = sorted(df['frame'].unique())
    records = []
    tracks = []
    next_id = 0

    # Estimate gravity from brownian motion
    brownian_trajectories = brownian(df)
    g = utils.gravity(brownian_trajectories, 10)

    for f in frames:
        detections = df[df['frame'] == f].to_numpy()  # [frame, x, y, r]

        # --- Predict all IMMs ---
        preds = []
        for t in tracks:
            t.imm.predict()
            preds.append(t.imm.x[:2, 0])
        preds = np.array(preds) if preds else np.zeros((0, 2))

        matched_rows, matched_cols = set(), set()

        # --- Data association ---
        if len(detections) > 0 and len(tracks) > 0:
            cost = np.linalg.norm(preds[:, None, :] - detections[None, :, 1:3], axis=2)
            cost[cost > max_distance] = 1e6  # disallow far matches
            row_ind, col_ind = linear_sum_assignment(cost)

            for r_idx, c_idx in zip(row_ind, col_ind):
                if cost[r_idx, c_idx] > max_distance:
                    continue
                meas = np.concatenate([detections[c_idx, 1:3], [g]])
                tracks[r_idx].imm.update(meas)
                pos = tracks[r_idx].imm.x[:2, 0]
                
                state = int(np.argmax(tracks[r_idx].imm.mu))

                records.append({
                    'frame': f,
                    'x': float(pos[0]), 'y': float(pos[1]),
                    'r': detections[c_idx, 3],
                    'particle': tracks[r_idx].id,
                    'state': state
                })
                matched_rows.add(r_idx)
                matched_cols.add(c_idx)
                tracks[r_idx].age = 0

        # --- Unmatched detections → new tracks (up to n_balls) ---
        for c_idx, det in enumerate(detections):
            if c_idx in matched_cols:
                continue
            if len(tracks) < n_balls:
                imm = make_imm(dt, g, meas_var, process_var)
                imm.x[:2, 0] = det[1:3]
                imm.x[2:, 0] = 0.0
                track = Track(imm, next_id)
                tracks.append(track)
                next_id += 1

        # --- Unmatched tracks → keep predicting ---
        for r_idx, t in enumerate(tracks):
            if r_idx in matched_rows:
                continue
            pos = t.imm.x[:2, 0]
            state = int(np.argmax(tracks[r_idx].imm.mu))
            records.append({
                'frame': f,
                'x': float(pos[0]), 'y': float(pos[1]),
                'r': np.nan,
                'particle': t.id,
                'state' : state
            })
            t.age += 1

    return pd.DataFrame(records)