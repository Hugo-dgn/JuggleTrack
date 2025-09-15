import cv2
import numpy as np
import tensorflow as tf
from gridmodel import GridModel
from ball_tracker import BallTracker

def configure_gpu():
    """Configures TensorFlow to use GPU memory growth."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e: print(e)

def get_tracking_data(video_path, model_path, n_balls):
    """
    Extracts raw trajectories for balls and hands without outlier filtering.
    """
    print("\n--- Extracting Trajectories ---")
    grid_model = GridModel(model_path, nBalls=n_balls, flip=False, postprocess=True)
    ball_tracker = BallTracker()
    tracking_data = {'balls': [[] for _ in range(n_balls)], 'left_hand_pos': [], 'right_hand_pos': []}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_dim = min(frame_width, frame_height)

    print(f"Processing {total_frames} frames...")
    
    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        if w > h: tocrop = int((w - h) / 2); cropped_frame = frame[:, tocrop:tocrop + h]
        elif h > w: tocrop = int((h - w) / 2); cropped_frame = frame[tocrop:tocrop + w, :]
        else: cropped_frame = frame
        if cropped_frame.shape[0] != output_dim:
             cropped_frame = cv2.resize(cropped_frame, (output_dim, output_dim))

        balls_and_hands = grid_model.predict(cropped_frame.copy())
        
        scale_factor = output_dim / 256.0
        rhand = (balls_and_hands["rhand"] * scale_factor).astype(int)
        lhand = (balls_and_hands["lhand"] * scale_factor).astype(int)
        detected_balls = (balls_and_hands["balls"] * scale_factor)
        
        tracked_balls = ball_tracker.update(detected_balls)

        if len(tracked_balls) == n_balls:
            for i in range(n_balls): tracking_data['balls'][i].append(tuple(tracked_balls[i]))
        tracking_data['left_hand_pos'].append(tuple(lhand))
        tracking_data['right_hand_pos'].append(tuple(rhand))
        
        if (frame_count + 1) % 100 == 0:
            print(f"  Processed {frame_count + 1} / {total_frames} frames...")

    print("\n--- Data Extraction Complete ---")
    cap.release()
    return tracking_data

def analyze_ball_states(tracking_data, hand_zone_dims=(120, 60)):
    """Analyzes trajectory data to determine when each ball is in a hand."""
    print("\n--- Analyzing Ball States (In Hand / In Flight) ---")
    n_balls = len(tracking_data['balls'])
    n_frames = min(len(tracking_data['left_hand_pos']), len(tracking_data['balls'][0]))
    analysis = [[] for _ in range(n_balls)]
    half_width, half_height = hand_zone_dims[0] / 2, hand_zone_dims[1] / 2

    for frame_idx in range(n_frames):
        left_pos, right_pos = np.array(tracking_data['left_hand_pos'][frame_idx]), np.array(tracking_data['right_hand_pos'][frame_idx])
        for ball_idx in range(n_balls):
            ball_pos = np.array(tracking_data['balls'][ball_idx][frame_idx])
            is_in_left = (abs(ball_pos[0] - left_pos[0]) < half_width) and (abs(ball_pos[1] - left_pos[1]) < half_height)
            is_in_right = (abs(ball_pos[0] - right_pos[0]) < half_width) and (abs(ball_pos[1] - right_pos[1]) < half_height)
            state = 'in_flight'
            if is_in_left and is_in_right: state = 'left_hand' if np.linalg.norm(ball_pos - left_pos) <= np.linalg.norm(ball_pos - right_pos) else 'right_hand'
            elif is_in_left: state = 'left_hand'
            elif is_in_right: state = 'right_hand'
            analysis[ball_idx].append(state)
    return analysis

def filter_short_hand_states(ball_states):
    """
    Post-processes the state analysis to remove "blips" where a ball is
    only considered in a hand for a single consecutive frame.
    """
    print("\n--- Filtering Single-Frame Hand Detections ---")
    filtered_analysis = []
    for states in ball_states:
        filtered_states = states[:] # Make a copy
        n_frames = len(states)
        for i in range(n_frames):
            if states[i] != 'in_flight':
                prev_state = states[i-1] if i > 0 else 'in_flight'
                next_state = states[i+1] if i < n_frames - 1 else 'in_flight'
                if prev_state == 'in_flight' and next_state == 'in_flight':
                    filtered_states[i] = 'in_flight' # Overwrite the blip
        filtered_analysis.append(filtered_states)
    return filtered_analysis