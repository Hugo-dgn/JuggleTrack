import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def track(all_circles):
    trajectories = []
    for idx, circles in enumerate(all_circles):
        if len(trajectories) == 0:
            for circle in circles:
                point = list(circle) + [idx]
                trajectories.append([point])
        else:
            new_pos = [(x, y) for x, y, _ in circles]
            old_circles = [t[-1] for t in trajectories]
            old_pos = [(x, y) for x, y, _, _ in old_circles]
            
            if len(new_pos) == 0:
                continue
            
            new_pos = np.array(new_pos)
            old_pos = np.array(old_pos)
            distances = cdist(new_pos, old_pos, metric='euclidean')
            row_ind, col_ind = linear_sum_assignment(distances)

            for i, j in zip(row_ind, col_ind):
                point = list(circles[i]) + [idx]
                trajectories[j].append(point)
    
    fill_trajectories = []
    n = len(all_circles)
    
    for t in trajectories:
        fill_trajectory = []
        i = 0
        for j in range(n):
            if i >= len(t):
                fill_trajectory.append((None, None))
            else:
                x, y, r, idx = t[i]
                if idx == j:
                    fill_trajectory.append((x, y))
                    i += 1
                else:
                    fill_trajectory.append((None, None))
        
        fill_trajectories.append(fill_trajectory)
        
        
    return np.array(fill_trajectories)
            