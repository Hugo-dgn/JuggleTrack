import cv2
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd

def compute_edges(frame):
    edges_channels = []

    for i in range(frame.shape[-1]):  # B, G, R channels
        channel = frame[:,:,i]
        edges = cv2.Canny(channel, 100, 150)
        edges_channels.append(edges.astype(np.float32)**2)  # square edges to emphasize strong edges

    # Combine channels using sqrt of sum of squares
    edges_combined = np.sqrt(np.sum(edges_channels, axis=0))
    edges_combined = np.uint8(np.clip(edges_combined, 0, 255))
    return edges_combined

def hough(path):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=True)
    all_records = []

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    person_segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
    
    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = person_segment.process(frame_rgb)
        person_mask = results.segmentation_mask
        _, person_mask = cv2.threshold(person_mask, 0.5, 255, cv2.THRESH_BINARY)
        person_mask = (255 - person_mask).astype("uint8")
        
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        masked = cv2.bitwise_and(frame, frame, mask=fgmask)
        masked = cv2.bitwise_and(masked, masked, mask=person_mask)
        
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=40,
            param2=30,
            minRadius=5,
            maxRadius=20
        )

        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))  # shape (n_circles, 3)
            for x, y, r in circles:
                all_records.append({'frame': frame_idx, 'x': float(x), 'y': float(y), 'r': float(r)})
                cv2.circle(debug, (int(x), int(y)), int(r), (0, 0, 255), 2)  # red circle

        cv2.imshow("debug", debug)
        cv2.waitKey(1)
    cap.release()
    return pd.DataFrame(all_records)

def contours(path):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_circles = []   # list of circles for each frame


    for _ in tqdm(range(total_frames), desc="Processing frames"):
        frame_circles = []
        ret, frame = cap.read()
        if not ret:
            all_circles.append(frame_circles)
            break
        
        
        edges = compute_edges(frame)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        circular_contours = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 2000:  # filter too small or too large
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter**2)
            if circularity > 0.8:  # threshold for circular contours
                circular_contours.append(cnt)
                (x, y), r = cv2.minEnclosingCircle(cnt)
                circle = (int(x), int(y), int(r))
                frame_circles.append(circle)
        all_circles.append(frame_circles)
    
    cap.release()
    return all_circles

def rgb(path, colors, colors_std):
    """
    Detect blobs of given colors in video and return a DataFrame:
    columns=['frame','x','y','particle'].
    
    path: video path
    colors: mean HSV colors (clusters)
    colors_std: std of HSV colors (clusters)
    radius_factor: fraction of blob radius to use (for reference)
    """
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    records = []

    # Setup blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 255
    detector = cv2.SimpleBlobDetector_create(params)

    lowers = np.clip(colors - 5*colors_std, 0, 255)
    uppers = np.clip(colors + 5*colors_std, 0, 255)

    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        masked = cv2.bitwise_and(frame, frame, mask=fgmask)

        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros_like(fgmask, dtype=bool)

        for lower, upper in zip(lowers, uppers):
            color_mask = cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8))
            combined_mask = np.logical_or(combined_mask, color_mask > 0)

        combined_mask = (combined_mask.astype(np.uint8) * 255)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.GaussianBlur(combined_mask, (11,11), 0)
        _, combined_mask = cv2.threshold(combined_mask, 128, 255, cv2.THRESH_BINARY)

        keypoints = detector.detect(combined_mask)
        debug = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        #debug = frame
        # Save positions in DataFrame format
        for particle_id, kp in enumerate(keypoints):
            x, y = kp.pt
            r = kp.size / 2
            records.append({'frame': frame_idx, 'x': float(x), 'y': float(y), 'r' : float(r), 'particle': particle_id})
            cv2.circle(debug, (int(x), int(y)), int(r), (0, 0, 255), 2)  # red circle

        cv2.imshow("Debug Detections", debug)
        cv2.waitKey(1)
    cap.release()
    return pd.DataFrame(records)
    

def get_colors(path, df, radius_factor=0.5, eps=15, min_samples=5):
    """
    Extract ball colors from video using positions in a DataFrame.
    
    path: video path
    df: DataFrame with columns ['frame','x','y','particle']
    radius_factor: fraction of radius to use for color patch
    eps, min_samples: DBSCAN parameters
    """
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ball_colors = []

    # Group detections by frame
    frame_groups = df.groupby('frame')

    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if frame_idx not in frame_groups.groups:
            continue  # no detections this frame

        detections = frame_groups.get_group(frame_idx)

        for _, row in detections.iterrows():
            x, y, r = int(row['x']), int(row['y']), row['r']
            r = int(r * radius_factor)

            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            patch = hsv[mask == 255]

            if patch.size > 0:
                mean_color = patch.mean(axis=0)
                ball_colors.append(mean_color)

    cap.release()
    ball_colors = np.array(ball_colors)

    # DBSCAN clustering of colors
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(ball_colors)
    labels = db.labels_

    # Optional 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ball_colors[:, 0], ball_colors[:, 1], ball_colors[:, 2], c=labels)
    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')
    plt.show()

    # Compute mean and std for each cluster
    colors = []
    colors_std = []
    for i in np.unique(labels):
        if i == -1:
            continue
        cluster = ball_colors[labels == i]
        colors.append(np.mean(cluster, axis=0))
        colors_std.append(np.std(cluster, axis=0))

    colors = np.array(colors)
    colors_std = np.array(colors_std)

    return colors, colors_std

def hands(video_path: str) -> pd.DataFrame:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
    
    hand_landmarks = {
            mp_pose.PoseLandmark.LEFT_INDEX,
            mp_pose.PoseLandmark.RIGHT_INDEX,
    }

    data = []
    frame_idx = 0

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                h, w, _ = frame.shape
                
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)

                    if idx in [hl.value for hl in hand_landmarks]:
                        color = (0, 0, 255)  # red for hands
                    else:
                        color = (0, 255, 0)  # green for rest

                    cv2.circle(frame, (x, y), 5, color, -1)

                # Left index
                left_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
                x, y = int(left_index.x * w), int(left_index.y * h)
                data.append({"frame": frame_idx, "x": x, "y": y, "hand": 1})

                # Right index
                right_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
                x, y = int(right_index.x * w), int(right_index.y * h)
                data.append({"frame": frame_idx, "x": x, "y": y, "hand": 0})
            
            cv2.imshow("Pose", frame)

            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
                

            frame_idx += 1
            pbar.update(1)  # update progress bar

    cap.release()
    df = pd.DataFrame(data)
    return df
