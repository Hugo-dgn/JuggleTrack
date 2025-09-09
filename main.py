import pandas as pd
import numpy as np
import os
from pathlib import Path

import segment
import tracking

path = Path("dataset/videos/ss_64x_id881.MP4")
name = Path(path.stem).with_suffix(".csv")
output = "results" / name
output.parent.mkdir(exist_ok=True)

all_circles = segment.hough(path)
colors, colors_std = segment.get_colors(path, all_circles)
all_circles = segment.rgb(path, colors, colors_std)
trajectories = tracking.track(all_circles)

x = np.transpose(trajectories, (1, 0, 2))
x = x.reshape(x.shape[0], -1)
df = pd.DataFrame(x)
nan_ratio = df.isna().mean()
almost_nan_cols = nan_ratio[nan_ratio > 0.9]
df = df.drop(columns=almost_nan_cols.index)
print(f"{df.shape[1] // 2} balls detected")
df.to_csv(output, index=False)
