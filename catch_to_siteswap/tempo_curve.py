import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tempo_curve(events, win=7, method="median"):
    """
    Construit une courbe de tempo locale (BPM en fonction du temps).

    - events: [{"time": float, "ball_id": ...}] triés par time
    - win: fenêtre pour lisser les intervalles (odd recommandé)
    - method: "median" ou "mean" pour le lissage

    Retourne:
      - t_mid: positions temporelles (centre des intervalles)
      - bpm_local: BPM estimé à ces positions
    """
    times = np.array([e["time"] for e in events])
    if len(times) < 2:
        return [], []

    dt = np.diff(times)
    series = pd.Series(dt)
    if method == "median":
        T_local = series.rolling(win, center=True, min_periods=1).median().to_numpy()
    else:
        T_local = series.rolling(win, center=True, min_periods=1).mean().to_numpy()

    bpm_local = 60.0 / T_local
    t_mid = (times[:-1] + times[1:]) / 2.0

    return t_mid, bpm_local


def plot_tempo_curve(events, win=7, method="median", ax=None):
    """
    Trace la courbe de tempo local (BPM) en fonction du temps avec matplotlib.
    """
    t_mid, bpm_local = tempo_curve(events, win=win, method=method)
    if len(t_mid) == 0:
        print("Pas assez d'événements pour estimer le tempo.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t_mid, bpm_local, marker="o", linestyle="-", label="Tempo local")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Tempo (BPM)")
    ax.set_title("Courbe de tempo de la prestation")
    ax.legend()
    plt.show()
