import cv2
import numpy as np

# Define a set of distinct colors for drawing
TRACKING_COLORS = [
    (255, 0, 0),   # Blue
    (0, 255, 255), # Yellow
    (255, 0, 255), # Magenta
    (0, 165, 255), # Orange
    (255, 255, 0), # Cyan
    (128, 0, 128)  # Purple
]
LEFT_HAND_COLOR = (0, 0, 255)   # Red
RIGHT_HAND_COLOR = (0, 255, 0)  # Green
IN_FLIGHT_COLOR = (255, 255, 255) # White

def draw_stateful_annotations(frame, lhand_pos, rhand_pos, ball_positions, ball_states):
    """
    Draws annotations on a frame, coloring balls based on their analyzed state
    (in left hand, in right hand, or in flight).
    """
    # Draw the hands first
    cv2.line(frame, (rhand_pos[0]-15, rhand_pos[1]), (rhand_pos[0]+15, rhand_pos[1]), RIGHT_HAND_COLOR, 4)
    cv2.line(frame, (lhand_pos[0]-15, lhand_pos[1]), (lhand_pos[0]+15, lhand_pos[1]), LEFT_HAND_COLOR, 4)

    # Draw each ball based on its state
    for i, pos in enumerate(ball_positions):
        state = ball_states[i]
        
        # Base color is the unique tracking color
        track_color = TRACKING_COLORS[i % len(TRACKING_COLORS)]
        
        # Border color indicates the state
        if state == 'left_hand':
            border_color = LEFT_HAND_COLOR
        elif state == 'right_hand':
            border_color = RIGHT_HAND_COLOR
        else: # 'in_flight'
            border_color = IN_FLIGHT_COLOR

        # Draw a thicker border to show the state, and a thinner inner circle with the track color
        cv2.circle(frame, tuple(pos), 15, border_color, 4)
        cv2.circle(frame, tuple(pos), 11, track_color, -1) # Filled inner circle
        
        # Draw the ball's unique ID number
        cv2.putText(frame, str(i), (pos[0] - 7, pos[1] + 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
