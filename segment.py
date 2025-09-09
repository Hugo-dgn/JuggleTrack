import cv2
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import DBSCAN
import numpy as np

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
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    all_circles = []   # list of circles for each frame


    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        masked = cv2.bitwise_and(frame, frame, mask=fgmask)
        
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=30,
            param2=30,
            minRadius=0,
            maxRadius=10
        )

        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))
        else:
            circles = []

        all_circles.append(circles)
    
    cap.release()
    return all_circles

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
            if circularity > 0.7:  # threshold for circular contours
                circular_contours.append(cnt)
                (x, y), r = cv2.minEnclosingCircle(cnt)
                circle = (int(x), int(y), int(r))
                frame_circles.append(circle)
        all_circles.append(frame_circles)
    
    cap.release()
    return all_circles

def rgb(path, colors, colors_std):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    all_circles = []   # list of circles for each frame
    
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = False
    params.minArea = 1   # adjust for your blobs
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 255

    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)


    for _ in tqdm(range(total_frames), desc="Processing frames"):
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
        lowers = colors - 4*colors_std
        uppers = colors + 4*colors_std
        
        lowers = np.clip(lowers, 0, 255)
        uppers = np.clip(uppers, 0, 255)
        
        mask = np.zeros_like(fgmask, dtype=bool)
        
        for lower, upper in zip(lowers, uppers):
            color_mask = cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8))
            mask = np.logical_or(mask, color_mask > 0)
        
        mask = mask.astype(np.uint8) * 255
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        keypoints = detector.detect(mask)
        
        circles = []
        for kp in keypoints:
            x, y = kp.pt                # floating-point center
            r = kp.size / 2.0           # radius
            circles.append((int(x), int(y), int(r)))
        all_circles.append(circles)
    
    cap.release()
    return all_circles
    

def get_colors(path, all_circles):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ball_colors = []
    for idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        circles = all_circles[idx]
        for (x, y, r) in circles:
            x, y, r = int(x), int(y), int(r)
            r = int(r/2)
            patch = hsv[max(0, y-r):y+r, max(0, x-r):x+r]

            if patch.size > 0:
                mean_color = patch.mean(axis=(0,1))
                ball_colors.append(mean_color)
    ball_colors = np.array(ball_colors)
    db = DBSCAN(eps=20, min_samples=5).fit(ball_colors)
    labels = db.labels_
    colors = []
    colors_std = []
    for i in np.unique(labels):
        colors.append(np.mean(ball_colors[labels == i], axis=0))
        colors_std.append(np.std(ball_colors[labels == i], axis=0))
    colors = np.array(colors)
    colors_std = np.array(colors_std)
    return colors, colors_std