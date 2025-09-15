import argparse
import cv2
import numpy as np

from cnn_detections.juggling_processor import (
    configure_gpu,
    get_tracking_data,
    analyze_ball_states,
    filter_short_hand_states,
)
from cnn_detections.drawingutils import draw_stateful_annotations


def create_annotated_video(video_path, output_path, tracking_data, ball_states):
    """
    Creates the final annotated video using the analyzed states.
    This function reads the original video and draws the processed data on top.
    """
    print("\n--- Creating Annotated Video ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get video properties for the writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Ensure we don't try to read more frames than we have data for
    total_frames = min(len(tracking_data["left_hand_pos"]), len(ball_states[0]))

    output_dim = min(frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_dim, output_dim))

    print(f"Rendering {total_frames} frames to {output_path}...")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Ran out of video frames at index {frame_idx}. Stopping.")
            break

        # Crop the frame to be square, same as during processing
        h, w, _ = frame.shape
        if w > h:
            tocrop = int((w - h) / 2)
            cropped_frame = frame[:, tocrop : tocrop + h]
        elif h > w:
            tocrop = int((h - w) / 2)
            cropped_frame = frame[tocrop : tocrop + w, :]
        else:
            cropped_frame = frame

        if cropped_frame.shape[0] != output_dim:
            cropped_frame = cv2.resize(cropped_frame, (output_dim, output_dim))

        # Gather all the necessary data for this specific frame
        lhand_pos = tracking_data["left_hand_pos"][frame_idx]
        rhand_pos = tracking_data["right_hand_pos"][frame_idx]
        current_ball_positions = [
            tracking_data["balls"][b][frame_idx] for b in range(len(ball_states))
        ]
        current_ball_states = [
            ball_states[b][frame_idx] for b in range(len(ball_states))
        ]

        # Draw the stateful annotations on the frame
        draw_stateful_annotations(
            cropped_frame,
            lhand_pos,
            rhand_pos,
            current_ball_positions,
            current_ball_states,
        )

        # Write the annotated frame to the output video
        writer.write(cropped_frame)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Rendered {frame_idx + 1} / {total_frames} frames...")

    print("\n--- Annotated Video Saved Successfully ---")
    cap.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a state-annotated video of a juggling performance."
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the input video file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../grid_models/grid_model_submovavg_64x64.h5",
        help="Path to the pre-trained grid model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_visuals.mp4",
        help="Path to save the final annotated video.",
    )
    parser.add_argument(
        "--n_balls", type=int, default=3, help="Number of balls to detect."
    )

    args = parser.parse_args()

    # Configure TensorFlow/GPU
    configure_gpu()

    # --- FULL PIPELINE EXECUTION FOR VISUALIZATION ---
    # Step 1: Get the raw tracking data
    tracking_data = get_tracking_data(args.video, args.model, args.n_balls)

    if tracking_data:
        # Step 2: Perform the initial state analysis
        raw_ball_states = analyze_ball_states(tracking_data)

        # Step 3: Filter out single-frame "blips" to get clean states
        filtered_states = filter_short_hand_states(raw_ball_states)

        # Step 4: Create the final annotated video using the clean data and states
        create_annotated_video(args.video, args.output, tracking_data, filtered_states)

        print("\nVisualization complete.")
