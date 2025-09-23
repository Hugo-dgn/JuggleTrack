# Installation

All the requirements are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

# Download Example Video

Download the file **`ss_64x_id881.MP4`** from the [Juggling Dataset](https://sites.google.com/view/jugglingdataset) and place it in:

```
dataset/videos/ss_64x_id881.MP4
```

# Ball Detection

First, detect the positions of the balls in the video:

```bash
python main.py detect dataset/videos/ss_64x_id881.MP4
```

This will save the results in `results/detections`.

# Ball Tracking

Then, track the balls to obtain their trajectories:

```bash
python main.py track results/detections/ss_64x_id881.csv 5
```

Here, `5` is the number of balls.
The result will be saved in `results/trajectories`.

# Hand Tracking

To detect the positions of the hands:

```bash
python main.py hands dataset/videos/ss_64x_id881.MP4
```

This will save the results in `results/hands`.

# Throw/Catch Detection

Run the throw/catch detection step:

```bash
python main.py transitions results/trajectories/ss_64x_id881.csv
```

The results will be stored in `results/transitions`.

# Visualize Results

You can visualize the results with:

```bash
python main.py display results/trajectories/ss_64x_id881.csv --transitions 0
```

This will print the throw and catch events for the ball with ID `0`.

# General Command Forms

For any video file `<video_path>` and number of balls `<num_balls>`, the general usage is:

```bash
# Detect balls
python main.py detect <video_path>

# Track balls
python main.py track results/detections/<video_name>.csv <num_balls>

# Track hands
python main.py hands <video_path>

# Detect throw/catch transitions
python main.py transitions results/trajectories/<video_name>.csv

# Visualize trajectories (with optional transitions for a given ball ID)
python main.py display results/trajectories/<video_name>.csv --transitions <ball_id>
```
