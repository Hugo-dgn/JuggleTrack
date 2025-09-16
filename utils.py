import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def load_trajectories(path):
    df = pd.read_csv(path)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    idx = df["particle"].to_numpy()
    t = df["frame"].to_numpy()
    state = df["state"].to_numpy()
    return x, y, t, idx

def detections_to_dataframe(detections):
    """
    Convert a list of detections into a DataFrame for trackpy.
    
    detections: list of arrays, each (n_detections, 2)
                Example: detections[frame] = [[x1, y1], [x2, y2], ...]
    Returns: pandas DataFrame with columns ['frame', 'x', 'y']
    """
    rows = []
    for frame_idx, frame in enumerate(detections):
        for det in frame:
            rows.append({"frame": frame_idx, "x": det[0], "y": det[1]})
    return pd.DataFrame(rows)

def gravity(trajectories, sigma):
    ys = trajectories['y'].to_numpy()
    ids = trajectories['particle'].to_numpy()
    t = trajectories['frame']
    
    acc = []
    for i in np.unique(ids):
        x = t[ids == i]
        y = ys[ids == i]
        y = gaussian_filter1d(y, sigma)
        
        if len(y) > 30:
            dy = np.gradient(y, x)
            d2y = np.gradient(dy, x)
            acc.append(d2y)
    
    acc = np.concatenate(acc)
    g = acc[acc > 0].mean()
    return g