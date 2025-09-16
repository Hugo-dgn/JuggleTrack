from pathlib import Path
import segment
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import utils

ball_id = 1

def transitions(trajectorie_path, hands_path):

    x, y, t, idx = utils.load_trajectories(trajectorie_path)
    hands = pd.read_csv(hands_path)
    
    ball_ids = np.unique(idx)

    events = []

    for ball_id in ball_ids:
        ball_mask = idx == ball_id

        x_hands = hands['x'].to_numpy()
        y_hands = hands['y'].to_numpy()
        t_hands = hands['frame'].to_numpy()
        right = hands['hand'].to_numpy() == 0
        left = hands['hand'].to_numpy() == 1

        # --- Right hand
        right_catch_idx, right_throw_idx, right_time = hand_transition(
            x_hands, y_hands, t_hands, x, y, t, right, ball_mask
        )
        for ti, ci in zip(right_time, right_catch_idx):
            events.append({
                "frame_idx": ci,
                "event": 0,       # 0 = catch
                "hand": 0,        # 0 = right
                "ball_id": ball_id
            })
        for ti, th in zip(right_time, right_throw_idx):
            events.append({
                "frame_idx": th,
                "event": 1,       # 1 = throw
                "hand": 0,
                "ball_id": ball_id
            })

        # --- Left hand
        left_catch_idx, left_throw_idx, left_time = hand_transition(
            x_hands, y_hands, t_hands, x, y, t, left, ball_mask
        )
        for ti, ci in zip(left_time, left_catch_idx):
            events.append({
                "frame_idx": ci,
                "event": 0,
                "hand": 1,        # 1 = left
                "ball_id": ball_id
            })
        for ti, th in zip(left_time, left_throw_idx):
            events.append({
                "frame_idx": th,
                "event": 1,
                "hand": 1,
                "ball_id": ball_id
            })

    df_events = pd.DataFrame(events).sort_values(by="frame_idx").reset_index(drop=True)
    return df_events

def hand_transition(x_hands, y_hands, t_hands, x, y, t, hand_mask, ball_mask):
    mask1, mask2, time = common_time_masks(t_hands[hand_mask], t[ball_mask])

    hand_y = y_hands[hand_mask][mask1]
    ball_y = y[ball_mask][mask2]
    hand_x = x_hands[hand_mask][mask1]
    ball_x = x[ball_mask][mask2]

    distances = np.sqrt((hand_y - ball_y)**2 + (hand_x - ball_x)**2)

    hand_std_y = np.std(hand_y)
    hand_std_x = np.std(hand_x)
    hand_std = np.sqrt(hand_std_y**2 + hand_std_x**2)

    condition = np.zeros(len(distances))

    condition[distances > 0.1*hand_std] = 1
    condition[distances > 2*hand_std] = 2
    jumps = np.where(np.concatenate((condition[1:] != condition[:-1], [True])))[0]
    throw_pattern = np.array([0, 1, 2])
    catch_pattern = np.array([2, 1, 0])
    states = condition[jumps]
    catch_matches = np.all(states[np.arange(len(states)-2)[:, None] + np.arange(3)] == catch_pattern, axis=1)
    throw_matches = np.all(states[np.arange(len(states)-2)[:, None] + np.arange(3)] == throw_pattern, axis=1)

    catch_idx = jumps[np.where(catch_matches)[0] + 1]
    throw_idx = jumps[np.where(throw_matches)[0]]
    
    return catch_idx, throw_idx, time

def common_time_masks(t1: np.ndarray, t2: np.ndarray):
    common_times = np.intersect1d(t1, t2)
    mask1 = np.isin(t1, common_times)
    mask2 = np.isin(t2, common_times)
    return mask1, mask2, common_times

if __name__ == "__main__":
    labels_path = Path("results/trajectories/ss_64x_id881.csv")
    video_path = "dataset/videos" / Path(labels_path.stem).with_suffix(".MP4")

    cap = cv2.VideoCapture(video_path)

    x, y, t, idx = utils.load_trajectories(labels_path)
    hands =  pd.read_csv("hands.csv")

    ball_mask = idx == ball_id

    x_hands = hands['x'].to_numpy()
    y_hands = hands['y'].to_numpy()
    t_hands = hands['frame'].to_numpy()
    right = hands['hand'].to_numpy() == 0
    left = hands['hand'].to_numpy() == 1

    mask1, mask2, time = common_time_masks(t_hands[right], t[ball_mask])

    hand_y = y_hands[right][mask1]
    ball_y = y[ball_mask][mask2]
    hand_x = x_hands[right][mask1]
    ball_x = x[ball_mask][mask2]

    distances = np.sqrt((hand_y - ball_y)**2 + (hand_x - ball_x)**2)

    hand_std_y = np.std(hand_y)
    hand_std_x = np.std(hand_x)
    hand_std = np.sqrt(hand_std_y**2 + hand_std_x**2)

    condition = np.zeros(len(distances))

    condition[distances > 0.5*hand_std] = 1
    condition[distances > 2*hand_std] = 2
    jumps = np.where(np.concatenate((condition[1:] != condition[:-1], [True])))[0]
    throw_pattern = np.array([0, 1, 2])
    catch_pattern = np.array([2, 1, 0])
    states = condition[jumps]
    catch_matches = np.all(states[np.arange(len(states)-2)[:, None] + np.arange(3)] == catch_pattern, axis=1)
    throw_matches = np.all(states[np.arange(len(states)-2)[:, None] + np.arange(3)] == throw_pattern, axis=1)

    catch_idx = jumps[np.where(catch_matches)[0] + 1]

    throw_idx = jumps[np.where(throw_matches)[0]]

    plt.figure()
    plt.plot(time, ball_y, label="ball")
    plt.plot(time, hand_y, label="hand")
    plt.scatter(time[catch_idx], ball_y[catch_idx], color="red", label="catch")
    plt.scatter(time[throw_idx], ball_y[throw_idx], color="blue", label="throw")
    plt.legend()
    plt.show()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if current frame is a catch
        if frame_idx in catch_idx:
            print(f"Catch at frame {frame_idx}")
        
        mask = np.where(np.logical_and(t == frame_idx, ball_mask))[0]
        if len(mask) > 0:
            i = mask[0]
            
            p_x = x[i]
            p_y = y[i]
            cv2.circle(frame, (int(p_x), int(p_y)), 10, (0, 0, 255), -1)
        # Display frame
        cv2.imshow("Video", frame)
        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()