import argparse
import numpy as np

from cnn_detections.juggling_processor import (
    configure_gpu,
    get_tracking_data,
    analyze_ball_states,
    filter_short_hand_states,
)


def find_catch_events(
    video_path, model_path="../grid_models/grid_model_submovavg_64x64.h5", n_balls=3
):
    """
    Runs the full juggling analysis pipeline on a video file and returns a
    list of all detected catch events.

    This is the main high-level function that encapsulates the entire process.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the pre-trained grid model.
        n_balls (int): The number of balls in the video.

    Returns:
        list: A list of dictionaries, each representing a single catch event,
              sorted chronologically. Returns an empty list if processing fails.
    """
    print(f"--- Starting Full Analysis for: {video_path} ---")

    # --- PIPELINE EXECUTION ---
    # Step 1: Get the raw tracking data
    tracking_data = get_tracking_data(video_path, model_path, n_balls)
    if not tracking_data:
        print("Error: Failed to extract tracking data. Aborting.")
        return []

    # Step 2: Perform the initial state analysis
    raw_ball_states = analyze_ball_states(tracking_data)

    # Step 3: Filter out single-frame "blips"
    filtered_states = filter_short_hand_states(raw_ball_states)

    # Step 4: Find the final list of catch events from the clean states
    print("\n--- Finding Catch Events ---")
    catch_events = []
    # Iterate through each ball's state trajectory
    for ball_id, states in enumerate(filtered_states):
        # Iterate through frames, starting from the second frame
        for frame_idx in range(1, len(states)):
            prev_state = states[frame_idx - 1]
            current_state = states[frame_idx]

            # The condition for a catch: a transition from flight to a hand
            if prev_state == "in_flight" and current_state != "in_flight":
                hand_id = "left" if current_state == "left_hand" else "right"

                event = {
                    "catch_frame": frame_idx,
                    "ball_id": ball_id,
                    "hand_id": hand_id,
                }
                catch_events.append(event)

    # Sort events by frame number for a chronological report
    catch_events.sort(key=lambda x: x["catch_frame"])

    print("--- Analysis Complete ---")
    return catch_events


if __name__ == "__main__":
    # This block is now ONLY for handling the Command-Line Interface (CLI)
    parser = argparse.ArgumentParser(
        description="Analyze a juggling video to find all catch events."
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
        "--n_balls", type=int, default=3, help="Number of balls in the video."
    )

    args = parser.parse_args()

    # Configure the GPU
    configure_gpu()

    # Call the main function with arguments from the CLI
    final_catch_events = find_catch_events(args.video, args.model, args.n_balls)

    # --- FINAL REPORT ---
    # This section is for formatting and printing the output for the user
    print("\n--- CATCH EVENT REPORT ---")
    if not final_catch_events:
        print("No catch events were detected.")
    else:
        for event in final_catch_events:
            print(
                f"Frame {event['catch_frame']:<5} | Ball {event['ball_id']} was caught by the {event['hand_id']} hand."
            )
    print("--------------------------\n")
