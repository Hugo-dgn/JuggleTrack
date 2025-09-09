import pandas as pd

def load_trajectories(path):
    df = pd.read_csv(path)
    labels = df.to_numpy()
    labels = labels.reshape(len(labels), -1, 2)
    return labels