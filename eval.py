from pathlib import Path

import utils

labels_path = Path("results/ss_64x_id881.csv")
gt_labels_path = "dataset/labels" / Path(labels_path.stem).with_suffix(".csv")

labels = utils.load_trajectories(labels_path)
gt = utils.load_trajectories(gt_labels_path)

pass