import numpy as np
import cv2

def color_range(colors, colors_std):
    color_ranges = []

    for mean_bgr, std_bgr in zip(colors, colors_std):
        lower_bgr = np.maximum(mean_bgr - 5*std_bgr, 0)
        upper_bgr = np.minimum(mean_bgr + 5*std_bgr, 255)
        
        color_ranges.append((lower_bgr, upper_bgr))
    return color_ranges