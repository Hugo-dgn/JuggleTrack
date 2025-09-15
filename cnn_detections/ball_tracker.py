import numpy as np
from scipy.optimize import linear_sum_assignment

class BallTracker:
    """
    Tracks juggling balls using a multi-pass Hungarian assignment with a dynamic,
    adaptive gating mechanism and a staleness counter to handle missed detections
    robustly through a "coast-then-freeze" strategy.
    """
    def __init__(self, alpha=0.7, max_staleness=5):
        """
        Initializes the tracker.

        Args:
            alpha (float): The smoothing factor for velocity's EMA.
            max_staleness (int): The number of consecutive frames a track can be
                                 unmatched before its velocity is reset.
        """
        self.positions = None
        self.velocities = None
        self.ball_ids = None
        # State for each track
        self.staleness = None 
        self.max_staleness = max_staleness
        self.alpha = alpha

    def _calculate_cost_matrix(self, predicted_positions, new_detections):
        """Calculates the Euclidean distance between predicted positions and new detections."""
        num_predicted = len(predicted_positions)
        num_new = len(new_detections)
        cost_matrix = np.zeros((num_predicted, num_new))
        for i in range(num_predicted):
            for j in range(num_new):
                cost_matrix[i, j] = np.linalg.norm(predicted_positions[i] - new_detections[j])
        return cost_matrix

    def update(self, detected_balls):
        """
        Updates the tracker with a new set of detected balls using the
        full robust tracking pipeline.
        """
        # --- FIX IS HERE: Ensure input is float for calculations ---
        detected_balls = np.array(detected_balls, dtype=float)

        if self.positions is None:
            self.positions = detected_balls
            self.velocities = np.zeros_like(self.positions, dtype=float)
            self.ball_ids = np.arange(len(self.positions))
            self.staleness = np.zeros(len(self.positions), dtype=int)
            # Return as int for drawing, but internal state remains float
            return self.positions.astype(int)

        # --- Predict & Age Phase ---
        self.staleness += 1
        predicted_positions = self.positions + self.velocities

        # --- Gating & Assignment Phase ---
        num_tracks = len(self.positions)
        num_detections = len(detected_balls)
        
        # Keep a reference to the positions from the start of the frame for velocity calculation
        old_positions = self.positions.copy()
        
        if num_detections > 0:
            predicted_dist_moved = np.linalg.norm(self.velocities, axis=1)
            base_gates = np.maximum(2.5 * predicted_dist_moved, 100.0)

            staleness_multiplier = 1.0 + (self.staleness * 0.25)
            dynamic_gates = base_gates * staleness_multiplier

            cost_matrix = self._calculate_cost_matrix(predicted_positions, detected_balls)
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)

            for track_idx, det_idx in zip(track_indices, detection_indices):
                cost = cost_matrix[track_idx, det_idx]
                if cost < dynamic_gates[track_idx]:
                    self.positions[track_idx] = detected_balls[det_idx]
                    self.staleness[track_idx] = 0

        # --- Update Phase ---
        # The self.positions array now contains a mix of newly detected positions
        # and old positions (for unmatched tracks).
        
        velocity_this_frame = self.positions - old_positions
        
        smoothed_velocity = (self.alpha * self.velocities) + ((1 - self.alpha) * velocity_this_frame)
        self.velocities = smoothed_velocity
        
        # --- Coast & Freeze Logic ---
        for i in range(num_tracks):
            if self.staleness[i] > 0:
                if self.staleness[i] <= self.max_staleness:
                    # COAST: The += operation will now work because both arrays are floats.
                    self.positions[i] += self.velocities[i]
                else:
                    self.velocities[i] *= 0.5
                    if np.linalg.norm(self.velocities[i]) < 1.0:
                        self.velocities[i] = np.array([0.0, 0.0])

        # --- FIX IS HERE: Return as integer for drawing ---
        # The internal state (self.positions) remains float for precision,
        # but the output is converted to int as expected by OpenCV.
        return self.positions.astype(int)