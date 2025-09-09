import cv2
import numpy as np

path = "/home/hugo/Documents/JuggleTrack/dataset/videos/ss_64x_id881.MP4"
cap = cv2.VideoCapture(path)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

all_circles = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Foreground detection ----
    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, 5)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Apply mask to frame (keep only moving parts)
    masked = cv2.bitwise_and(frame, frame, mask=fgmask)

    # ---- Circle detection ----
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=100,
        param2=30,
        minRadius=5,
        maxRadius=20
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
    else:
        circles = []

    all_circles.append(circles)

    # ---- Visualization (for debugging) ----
    debug = masked
    for (x, y, r) in circles:
        cv2.circle(debug, (x, y), r, (0, 255, 0), 2)
        cv2.circle(debug, (x, y), 2, (0, 0, 255), 3)

    #cv2.imshow("Foreground Mask", fgmask)
    cv2.imshow("Detection", debug)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
