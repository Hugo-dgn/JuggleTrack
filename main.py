from pathlib import Path
import argparse
import pandas as pd
import cv2
import numpy as np
import state

import segment
import tracking

def detect(args):
    video_path = Path(args.video)
    output_folder = Path("results/detections")
    output_folder.mkdir(exist_ok=True, parents=True)
    output_file = output_folder / f"{video_path.stem}.csv"

    print(f"[INFO] Processing video: {video_path}")
    print("[INFO] Running Hough detection...")
    detections = segment.hough(video_path)

    print("[INFO] Extracting colors...")
    colors, colors_std = segment.get_colors(video_path, detections)

    print("[INFO] Running color-based RGB detection...")
    detections_df = segment.rgb(video_path, colors, colors_std)

    print(f"[INFO] Saving results to {output_file}")
    detections_df.to_csv(output_file, index=False)
    print("[DONE]")

def track(args):
    detections_path = Path(args.data)
    if not detections_path.exists():
        print(f"[ERROR] Detections file not found: {detections_path}")
        return

    output_folder = Path("results/trajectories")
    output_folder.mkdir(exist_ok=True, parents=True)
    output_file = output_folder / f"{detections_path.stem}.csv"

    print(f"[INFO] Loading detections: {detections_path}")
    detections = pd.read_csv(detections_path)

    print("[INFO] Running tracking algorithm...")
    trajectories = tracking.kalman(detections, args.n_balls)

    print(f"[INFO] Saving trajectories to {output_file}")
    trajectories.to_csv(output_file, index=False)
    print("[DONE]")

def transitions(args):
    trajectories_path = Path(args.trajectories)
    if not trajectories_path.exists():
        print(f"[ERROR] Trajectories file not found: {trajectories_path}")
        return

    hands_dir = Path("results/hands")
    hands_stem = trajectories_path.stem  # e.g. "ss_64x_id881"
    
    hands_path = hands_dir / Path(hands_stem).with_suffix(".csv")
    if not hands_path.exists():
        raise Exception(f"No hands positions found : {hands_path}")
    
    df = state.transitions(trajectories_path, hands_path)
    
    output_dir = Path("results/transitions")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / Path(hands_stem).with_suffix(".csv")
    df.to_csv(output_file, index=False)

def display(args):
    labels_path = Path(args.data)
    if not labels_path.exists():
        print(f"[ERROR] Trajectories file not found: {labels_path}")
        return

    video_dir = Path("dataset/videos")
    video_stem = labels_path.stem  # e.g. "ss_64x_id881"

    # Look for any file with the same stem in the directory
    video_files = list(video_dir.glob(video_stem + ".*"))

    if video_files:
        video_path = video_files[0]  # first match
        print("Found video:", video_path)
    else:
        print("No matching video found")
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return
    
    if args.transitions is not None:
        transitions_dir = Path("results/transitions")
        transitions_file = transitions_dir / Path(labels_path.stem).with_suffix(".csv")
        transitions = pd.read_csv(transitions_file)
        mask = transitions["ball_id"] == args.transitions
        frames = transitions.loc[mask, "frame_idx"].to_numpy()
        hands = transitions.loc[mask, "hand"].to_numpy()
        events = transitions.loc[mask, "event"].to_numpy()

    cap = cv2.VideoCapture(str(video_path))
    trajectories = pd.read_csv(labels_path)

    n_balls = trajectories['particle'].nunique()
    colors = np.random.randint(0, 256, size=(n_balls, 3)).tolist()
    
    frame_data = trajectories[trajectories['particle'] == 3]
    x, y = frame_data[['x', 'y']].to_numpy().T

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = trajectories[trajectories['frame'] == idx]
        positions = frame_data[['x', 'y', 'r']].to_numpy()  # get radius from DF
        balls_id = frame_data['particle'].to_numpy()

        for ball_id, (x, y, r) in zip(balls_id, positions):
            if not np.isnan(r):
                x, y, r, ball_id = int(x), int(y), int(r), int(ball_id)
                cv2.circle(frame, (x, y), r, colors[ball_id], 2)
                cv2.putText(frame, str(ball_id), (x + r + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[ball_id], 2)
        
        if args.transitions is not None:
            if idx in frames:
                i = np.where(frames == idx)[0]
                hand = "left" if hands[i] else "right"
                event = "throw" if events[i] else "catch"
                
                print(f"{hand} {event}")

        cv2.imshow("results", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        idx += 1

    cap.release()
    cv2.destroyAllWindows()

def hands(args):
    video_path = Path(args.video)
    output_folder = Path("results/hands")
    output_folder.mkdir(exist_ok=True, parents=True)
    output_file = output_folder / f"{video_path.stem}.csv"

    print(f"[INFO] Processing video: {video_path}")
    print("[INFO] Running Hands detection...")
    detections = segment.hands(video_path)

    print(f"[INFO] Saving results to {output_file}")
    detections.to_csv(output_file, index=False)
    print("[DONE]")

def main():
    parser = argparse.ArgumentParser(description="Video processing pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'detect' subcommand
    parser_detect = subparsers.add_parser("detect", help="Detect balls in a video")
    parser_detect.add_argument("video", type=str, help="Path to the video file")
    parser_detect.set_defaults(func=detect)

    # 'track' subcommand
    parser_track = subparsers.add_parser("track", help="Track particles from detections CSV")
    parser_track.add_argument("data", type=str, help="Path to the detections CSV file")
    parser_track.add_argument("n_balls", type=int, help="Number of balls.")
    parser_track.set_defaults(func=track)
    
    # display subcommand
    parser_display = subparsers.add_parser("display", help="Display tracked particles on video")
    parser_display.add_argument("data", type=str, help="Path to the trajectories CSV file")
    parser_display.add_argument("--transitions", type=int, help="Print transition for the given ball id")
    parser_display.set_defaults(func=display)
    
    parser_hands = subparsers.add_parser("hands", help="Detect hands in a video")
    parser_hands.add_argument("video", type=str, help="Path to the video file")
    parser_hands.set_defaults(func=hands)
    
    parser_transitions = subparsers.add_parser("transitions", help="Detect catch and throws")
    parser_transitions.add_argument("trajectories", type=str, help="Path to the trajectories file")
    parser_transitions.set_defaults(func=transitions)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
