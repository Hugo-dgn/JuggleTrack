import argparse
import cv2

# Import all necessary functions from our library files
from cnn_detections.juggling_processor import (
    configure_gpu,
    get_tracking_data,
    analyze_ball_states,
    filter_short_hand_states,
    get_video_rotation
)
from cnn_detections.main_analyzer import find_catch_events
from cnn_detections.drawingutils import draw_stateful_annotations


def create_annotated_video(
    video_path, output_path, tracking_data, ball_states, catch_events, highlight_window=2, rotation_code=None
):
    """
    Creates the final annotated video, showing continuous state tracking and
    adding a special highlight effect for catch events.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output MP4 video.
        tracking_data (dict): The raw tracking data for hands and balls.
        ball_states (list): A list of state lists for each ball.
        catch_events (list): A list of detected catch event dictionaries.
        highlight_window (int): Number of frames before and after a catch to show the highlight.
        rotation_code (int, optional): The cv2 rotation code to apply to each frame. Defaults to None.
    """
    print("\n--- Creating Annotated Video with Catch Highlights ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # --- FIX 1: Determine correct output dimensions from a rotated frame ---
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video.")
        cap.release()
        return
    
    # Apply rotation before measuring dimensions to get the correct output size
    if rotation_code is not None:
        frame = cv2.rotate(frame, rotation_code)
        
    frame_height, frame_width, _ = frame.shape
    output_dim = min(frame_width, frame_height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Rewind the video capture to the beginning for the main rendering loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_dim, output_dim))

    # Ensure we don't process more frames than we have data for
    total_frames = min(len(tracking_data["left_hand_pos"]), len(ball_states[0]))
    
    highlights = {}
    for event in catch_events:
        frame = event["catch_frame"]
        for i in range(frame - highlight_window, frame + highlight_window + 1):
            if i < 0 or i >= total_frames:
                continue
            if i not in highlights:
                highlights[i] = []
            highlights[i].append(event)

    print(f"Rendering {total_frames} frames to {output_path}...")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Ran out of video frames at index {frame_idx}. Stopping.")
            break

        # Use the passed-in rotation_code for consistent orientation
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)
    
        # Crop the frame to be square
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

        # Layer 1: Draw highlights
        if frame_idx in highlights:
            overlay = cropped_frame.copy()
            for event in highlights[frame_idx]:
                if event["ball_id"] < len(tracking_data["balls"]) and frame_idx < len(tracking_data["balls"][event["ball_id"]]):
                    ball_pos = tracking_data["balls"][event["ball_id"]][frame_idx]
                    cv2.circle(overlay, center=tuple(ball_pos), radius=40, color=(0, 255, 255), thickness=-1)
            
            alpha = 0.4
            cv2.addWeighted(src1=overlay, alpha=alpha, src2=cropped_frame, beta=(1 - alpha), gamma=0, dst=cropped_frame)
            
            for event in highlights[frame_idx]:
                if event["ball_id"] < len(tracking_data["balls"]) and frame_idx < len(tracking_data["balls"][event["ball_id"]]):
                    ball_pos = tracking_data["balls"][event["ball_id"]][frame_idx]
                    text_pos = (ball_pos[0] + 25, ball_pos[1] - 25)
                    cv2.putText(
                        img=cropped_frame, text=str(event["ball_id"]), org=text_pos,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
                        color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA
                    )

        # Layer 2: Draw standard annotations
        lhand_pos = tracking_data["left_hand_pos"][frame_idx]
        rhand_pos = tracking_data["right_hand_pos"][frame_idx]
        current_ball_positions = [tracking_data["balls"][b][frame_idx] for b in range(len(ball_states))]
        current_ball_states = [ball_states[b][frame_idx] for b in range(len(ball_states))]

        draw_stateful_annotations(
            frame=cropped_frame,
            lhand_pos=lhand_pos,
            rhand_pos=rhand_pos,
            ball_positions=current_ball_positions,
            ball_states=current_ball_states,
        )

        writer.write(cropped_frame)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Rendered {frame_idx + 1} / {total_frames} frames...")

    print("\n--- Annotated Video Saved Successfully ---")
    cap.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a state-annotated video of a juggling performance with catch highlights."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--model", type=str, default="grid_models/grid_model_submovavg_64x64.h5", help="Path to the pre-trained grid model.")
    parser.add_argument("--output", type=str, default="output_visuals.mp4", help="Path to save the final annotated video.")
    parser.add_argument("--n_balls", type=int, default=3, help="Number of balls to detect.")
    parser.add_argument("--min_flight", type=int, default=6, help="Minimum flight frames for a SAME-HAND catch.")
    parser.add_argument("--highlight_window", type=int, default=2, help="Number of frames before and after a catch to apply the highlight.")
    args = parser.parse_args()

    configure_gpu()

    # --- FIX 2: Replicate the correct analysis pipeline to ensure data consistency ---

    # Determine rotation ONCE. This is the single source of truth for this script.
    rotation_code = get_video_rotation(args.video)

    # Step 1: Get tracking data using the correct rotation.
    tracking_data = get_tracking_data(args.video, args.model, args.n_balls, rotation_code=rotation_code)

    if tracking_data:
        # Steps 2 & 3: Process the correctly rotated tracking data.
        raw_ball_states = analyze_ball_states(tracking_data)
        filtered_states = filter_short_hand_states(raw_ball_states)
        
        # Step 4: Call the main analysis function to get the final list of catch events.
        catch_events = find_catch_events(
            video_path=args.video,
            model_path=args.model,
            n_balls=args.n_balls,
            min_flight_frames=args.min_flight
        )

        # Step 5: Create the video, passing the consistent data AND the rotation code.
        create_annotated_video(
            video_path=args.video,
            output_path=args.output,
            tracking_data=tracking_data,
            ball_states=filtered_states,
            catch_events=catch_events,
            highlight_window=args.highlight_window,
            rotation_code=rotation_code # Pass the code for consistent rendering
        )

        print("\nVisualization complete.")