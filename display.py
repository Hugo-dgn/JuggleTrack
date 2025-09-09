import cv2
import numpy as np
from pathlib import Path
import utils

labels_path = Path("results/ss_64x_id881.csv")
video_path = "dataset/videos" / Path(labels_path.stem).with_suffix(".MP4")

cap = cv2.VideoCapture(video_path)
labels = utils.load_trajectories(labels_path)

n = labels.shape[1]
colors = np.random.randint(0, 256, size=(n, 3)).tolist()

idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    r = 10
    for color, points in zip(colors, labels[idx]):
        x, y = points
        if not np.isnan(x) and not np.isnan(y):
            x, y, r = int(x), int(y), int(r)
            cv2.circle(frame, (x, y), r, color, 2)
    cv2.imshow("results", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    idx += 1

cap.release()
cv2.destroyAllWindows()