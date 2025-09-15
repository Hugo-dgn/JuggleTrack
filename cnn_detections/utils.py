import numpy as np

def getDistance(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points (x1, y1) and (x2, y2).
    This function uses NumPy and remains unchanged.
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

