import argparse
import numpy as np
# Import the functions from our library file
from juggling_processor import (
    configure_gpu,
    get_tracking_data,
    analyze_ball_states,
    filter_short_hand_states
)

def find_catch_events(filtered_ball_states, min_flight_frames=6):
    """
    Identifies catch events with context-aware validation.

    - A catch is always considered VALID if it's a hand-to-hand transfer
      (i.e., the catching hand is different from the last hand that held the ball).
    - A catch is only considered VALID for a same-hand throw if the ball was
      'in_flight' for at least `min_flight_frames`.

    Args:
        filtered_ball_states (list): The cleaned list of state lists.
        min_flight_frames (int): The minimum flight time for a SAME-HAND catch.

    Returns:
        list: A list of dictionaries, each representing a single valid catch event.
    """
    print(f"\n--- Finding Catch Events (min same-hand flight time: {min_flight_frames} frames) ---")
    catch_events = []
    
    # Iterate through each ball's state trajectory
    for ball_id, states in enumerate(filtered_ball_states):
        
        # State machine variables for each ball
        # NEW: Track which hand last held the ball ('left', 'right', or None)
        last_hand_id = None
        if states[0] == 'left_hand':
            last_hand_id = 'left'
        elif states[0] == 'right_hand':
            last_hand_id = 'right'
            
        throw_frame_index = 0 # The frame when the ball last became 'in_flight'

        # Iterate through frames, starting from the second frame
        for frame_idx in range(1, len(states)):
            prev_state = states[frame_idx - 1]
            current_state = states[frame_idx]

            # --- Update Last Known Hand ---
            # This is crucial: we must always know which hand was the last to hold the ball.
            if current_state == 'left_hand':
                last_hand_id = 'left'
            elif current_state == 'right_hand':
                last_hand_id = 'right'

            # --- Detect a THROW event ---
            if prev_state != 'in_flight' and current_state == 'in_flight':
                throw_frame_index = frame_idx
            
            # --- Detect a CATCH event ---
            elif prev_state == 'in_flight' and current_state != 'in_flight':
                catching_hand_id = 'left' if current_state == 'left_hand' else 'right'
                flight_duration = frame_idx - throw_frame_index
                
                # --- NEW CONTEXT-AWARE VALIDATION LOGIC ---
                is_valid_catch = False
                
                # Case 1: Hand-to-Hand Transfer. Always valid.
                # This also covers the very first catch where last_hand_id is None.
                if catching_hand_id != last_hand_id:
                    is_valid_catch = True
                
                # Case 2: Same-Hand Throw. Must have minimum flight time.
                elif flight_duration >= min_flight_frames:
                    is_valid_catch = True

                if is_valid_catch:
                    event = {
                        "catch_frame": frame_idx,
                        "ball_id": ball_id,
                        "hand_id": catching_hand_id
                    }
                    catch_events.append(event)
                # If neither condition is met, it's an invalid short "blip" to the same hand.

    # Sort events by frame number for a chronological report
    catch_events.sort(key=lambda x: x['catch_frame'])
    return catch_events

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze a juggling video to find all valid catch events.")
    parser.add_argument('--video', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--model', type=str, default='../grid_models/grid_model_submovavg_64x64.h5', help="Path to the pre-trained grid model.")
    parser.add_argument('--n_balls', type=int, default=3, help="Number of balls in the video.")
    # MODIFIED: Clarified the help text for this argument
    parser.add_argument('--min_flight', type=int, default=4, help="Minimum flight frames required for a SAME-HAND catch to be valid.")
    
    args = parser.parse_args()
    
    configure_gpu()
    
    tracking_data = get_tracking_data(args.video, args.model, args.n_balls)

    if tracking_data:
        raw_ball_states = analyze_ball_states(tracking_data)
        filtered_states = filter_short_hand_states(raw_ball_states)
        
        final_catch_events = find_catch_events(filtered_states, min_flight_frames=args.min_flight)
        
        print("\n--- CATCH EVENT REPORT ---")
        if not final_catch_events:
            print("No valid catch events were detected.")
        else:
            for event in final_catch_events:
                print(f"Frame {event['catch_frame']:<5} | Ball {event['ball_id']} was caught by the {event['hand_id']} hand.")
        print("--------------------------\n")