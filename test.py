import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Video input
cap = cv2.VideoCapture("dataset/videos/test.mp4")

# Define which landmarks are part of the hands
hand_landmarks = {
    mp_pose.PoseLandmark.LEFT_INDEX,
    mp_pose.PoseLandmark.RIGHT_INDEX,
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw keypoints
    if results.pose_landmarks:
        h, w, _ = frame.shape
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)

            if idx in [hl.value for hl in hand_landmarks]:
                color = (0, 0, 255)  # red for hands
            else:
                color = (0, 255, 0)  # green for rest

            cv2.circle(frame, (x, y), 5, color, -1)

    # Show video
    cv2.imshow("Pose", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

