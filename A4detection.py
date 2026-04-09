import cv2
import numpy as np
import time
import heapq # Added for A* Priority Queue
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

# =============================================================================
# HELPER CLASSES & FUNCTIONS FOR A* SEARCH
# =============================================================================

class Node:
    """Node class to store state for A* Search"""
    def __init__(self, corners, g, h):
        self.corners = corners  # List of coordinates e.g., [(x1,y1), (x2,y2)...]
        self.g = g              # Cost from start (number of edges)
        self.h = h              # Heuristic cost to goal
        
    def f(self):
        return self.g + self.h
        
    def __lt__(self, other):
        # Priority Queue will sort nodes based on lowest f(n)
        return self.f() < other.f()

def check_angle(p1, p2, p3):
    """Checks if the angle between 3 points is approximately 90 degrees."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return False
        
    dot_product = np.dot(v1, v2)
    cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    return 75 <= angle <= 105

def check_edge_strength(p1, p2, bw):
    """Calculates the confidence score of an edge based on pixel brightness."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    
    jarak = max(abs(x2 - x1), abs(y2 - y1))
    if jarak == 0: 
        return 0.0
        
    xs = np.clip(np.rint(np.linspace(x1, x2, jarak)).astype(int), 0, bw.shape[1]-1)
    ys = np.clip(np.rint(np.linspace(y1, y2, jarak)).astype(int), 0, bw.shape[0]-1)
    
    edge_vals = bw[ys, xs]
    confidence = np.mean(edge_vals) 
    return confidence

def heuristic_1(k):
    """H1: Blind heuristic based only on remaining corners."""
    return float(4 - k)

def heuristic_2(k, corners, bw):
    """H2: Smart heuristic utilizing visual edge evidence."""
    if k < 2:
        return float(4 - k)
    
    # Check the confidence of the most recently formed edge
    p1, p2 = corners[-2], corners[-1]
    confidence = check_edge_strength(p1, p2, bw)
    
    # Formula: (4 - k) * (1 - confidence score)
    return (4 - k) * (1.0 - confidence)

def likelihoodA4(rect, bw, weightA4Prior=0.3, neighborhood=0):
    """Port of the MATLAB likelihoodA4 function."""
    if rect.shape != (4, 2):
        raise ValueError("rect should be 4x2 array with corners in (x,y) form.")

    image_height, image_width = bw.shape
    p_edge = 0.0

    for i in range(4):
        pt1 = rect[i]
        pt2 = rect[(i + 1) % 4]

        if np.allclose(pt1, pt2):
            return 0.0

        x1, y1 = pt1
        x2, y2 = pt2

        num_steps = int(2 * max(image_height, image_width))
        if num_steps < 1:
            num_steps = 1

        xs = np.linspace(x1, x2, num=num_steps, endpoint=True)
        ys = np.linspace(y1, y2, num=num_steps, endpoint=True)

        xs = np.rint(xs).astype(int)
        ys = np.rint(ys).astype(int)

        line_points = np.column_stack((ys, xs))
        line_points = np.unique(line_points, axis=0)

        if neighborhood > 0:
            expanded = []
            for (row, col) in line_points:
                if neighborhood == 1:
                    neighbors = [
                        (row, col),
                        (row + 1, col), (row - 1, col),
                        (row, col + 1), (row, col - 1)
                    ]
                else: 
                    neighbors = [
                        (row, col),
                        (row + 1, col),   (row - 1, col),
                        (row, col + 1),   (row, col - 1),
                        (row + 1, col+1), (row - 1, col+1),
                        (row + 1, col-1), (row - 1, col-1)
                    ]
                expanded.extend(neighbors)
            line_points = np.unique(np.array(expanded), axis=0)

        mask_in_bounds = (
            (line_points[:, 0] >= 0) & (line_points[:, 0] < image_height) &
            (line_points[:, 1] >= 0) & (line_points[:, 1] < image_width)
        )
        line_points = line_points[mask_in_bounds]

        if len(line_points) == 0:
            return 0.0

        edge_vals = bw[line_points[:, 0], line_points[:, 1]]
        fraction_on_edge = np.sum(edge_vals) / float(len(edge_vals))
        p_edge += fraction_on_edge

    p_edge /= 4.0

    v1 = rect[0] - rect[1]
    v2 = rect[1] - rect[2]
    v3 = rect[2] - rect[3]
    v4 = rect[3] - rect[0]

    normv1 = np.linalg.norm(v1)
    normv2 = np.linalg.norm(v2)
    normv3 = np.linalg.norm(v3)
    normv4 = np.linalg.norm(v4)

    if normv1 > normv2:
        ratio1 = normv1 / (normv2 + 1e-6)
        ratio2 = normv3 / (normv4 + 1e-6)
    else:
        ratio1 = normv2 / (normv1 + 1e-6)
        ratio2 = normv4 / (normv3 + 1e-6)

    ratio = (ratio1 + ratio2) / 2.0

    mu = 297.0 / 210.0
    sigma = 0.5
    q = np.exp(-(ratio - mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    p = (1 - weightA4Prior) * p_edge + weightA4Prior * q
    return p


# =============================================================================
# MAIN FUNCTION WITH MODE SWITCH
# =============================================================================

def detect_a4_main(filepath, mode="h2"):
    """
    Detects an A4 rectangle using A* Search.
    mode parameters: "h0" (BFS), "h1" (A* with H1), "h2" (A* with H2).
    """
    thresholdOfBeingA4  = 0.8     
    weightA4Prior       = 0.3     
    neighborhood        = 0       
    nHoughPeaks         = 40      
    angle_h             = 75      
    angle_l             = 15      

    target_height = 2800
    target_width = 2100

    A4_WIDTH_CM  = 21.0
    A4_HEIGHT_CM = 29.7

    start_time = time.time()

    # --- Step 1 to 5: Image Pre-processing (Dr. Yasir's Code) ---
    originalImage = cv2.imread(filepath)
    if originalImage is None:
        raise IOError(f"Could not read image from {filepath}")

    greyImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    h, w = greyImage.shape
    if h < w:
        greyImage = np.rot90(greyImage, k=3)

    greyImage = cv2.resize(greyImage, (target_width, target_height), interpolation=cv2.INTER_AREA)

    grey_float = greyImage.astype(np.float32)
    gaussian1 = cv2.GaussianBlur(grey_float, (21, 21), 20)
    gaussian2 = cv2.GaussianBlur(grey_float, (21, 21), 35)
    dogFilterImage = gaussian1 - gaussian2

    dog_min = dogFilterImage.min()
    dog_max = dogFilterImage.max()
    norm_dog = 255.0 * (dogFilterImage - dog_min) / (dog_max - dog_min + 1e-8)
    norm_dog = norm_dog.astype(np.uint8)

    _, binaryImage = cv2.threshold(norm_dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaryImage = cv2.erode(binaryImage, np.ones((3, 3), np.uint8))
    bin_bool = (binaryImage > 0)

    # --- Step 6 & 7: Hough Transform ---
    tested_angles = np.deg2rad(np.arange(-90, 90, 0.5))
    hspace, angles, dists = hough_line(bin_bool, theta=tested_angles)

    h_peaks, theta_peaks, dist_peaks = hough_line_peaks(
        hspace, angles, dists, num_peaks=nHoughPeaks, threshold=0.1 * np.max(hspace)
    )

    deg_thetas = np.rad2deg(theta_peaks)
    keep_mask = []
    for i, th_deg in enumerate(deg_thetas):
        if ((th_deg >= -angle_h and th_deg <= -angle_l) or
            (th_deg >= angle_l  and th_deg <= angle_h)):
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    keep_mask = np.array(keep_mask, dtype=bool)
    theta_peaks = theta_peaks[keep_mask]
    dist_peaks  = dist_peaks[keep_mask]

    # --- Step 8: Find Intersections ---
    intersection_list = []
    for i in range(len(dist_peaks)):
        rho1, theta1 = dist_peaks[i], theta_peaks[i]
        for j in range(i+1, len(dist_peaks)):
            rho2, theta2 = dist_peaks[j], theta_peaks[j]

            if np.isclose(theta1, theta2, atol=1e-6):
                continue

            A = np.array([[np.cos(theta1), np.sin(theta1)],
                          [np.cos(theta2), np.sin(theta2)]])
            b = np.array([rho1, rho2])
            try:
                sol = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue

            xSol, ySol = sol
            if not (target_width * 0.15 < xSol < target_width * 0.85): continue
            if not (target_height * 0.05 < ySol < target_height * 0.80): continue

            intersection_list.append((int(round(xSol)), int(round(ySol))))

    xy_original = np.array(intersection_list)
    if len(xy_original) == 0:
        return False, np.zeros((4,2)), 0.0, 0.0, 0, time.time() - start_time

    # Remove duplicates
    xy_unique = np.unique(xy_original, axis=0)
    unique_candidates = [tuple(pt) for pt in xy_unique]
    
    # --- Step 9: A* SEARCH IMPLEMENTATION ---
    search_start_time = time.time()
    
    found = False
    bestScore = 0.0
    bestA4 = np.zeros((4, 2), dtype=float)
    nodes_expanded = 0
    
    open_set = []
    
    # Determine initial h based on selected mode
    if mode == "h1":
        h_init = heuristic_1(0)
    elif mode == "h2":
        h_init = heuristic_2(0, [], bin_bool)
    else:
        h_init = 0.0 # BFS
        
    start_node = Node(corners=[], g=0, h=h_init)
    heapq.heappush(open_set, start_node)
    
    while open_set:
        current_node = heapq.heappop(open_set)
        nodes_expanded += 1
        k = len(current_node.corners)
        
        # Goal State Checking (k=4)
        if k == 4:
            rect = np.array(current_node.corners, dtype=float)
            score = likelihoodA4(rect, bin_bool, weightA4Prior, neighborhood)
            
            if score > bestScore:
                bestScore = score
                bestA4 = rect.copy()
                
            # Early termination if the score is excellent
            if bestScore >= thresholdOfBeingA4:
                found = True
                break
                
            continue # Try next node in the queue
            
        # State Expansion (Adding new corners)
        for pt in unique_candidates:
            if pt in current_node.corners:
                continue
                
            new_corners = current_node.corners.copy()
            new_corners.append(pt)
            new_k = len(new_corners)
            
            # Distance constraint (k=2)
            if new_k == 2:
                jarak = np.linalg.norm(np.array(new_corners[0]) - np.array(new_corners[1]))
                if jarak < 50.0: 
                    continue 
                    
            # Geometric & Ratio constraints (k=3)
            if new_k == 3:
                p1, p2, p3 = new_corners[0], new_corners[1], new_corners[2]
                if not check_angle(p1, p2, p3):
                    continue
                    
                L1 = np.linalg.norm(np.array(p1) - np.array(p2))
                L2 = np.linalg.norm(np.array(p2) - np.array(p3))
                ratio_bounds = max(L1, L2) / (min(L1, L2) + 1e-6)
                if ratio_bounds < 1.2 or ratio_bounds > 1.6: 
                    continue
                    
            # Final Geometric constraint (k=4)
            if new_k == 4:
                p1, p2, p3, p4 = new_corners[0], new_corners[1], new_corners[2], new_corners[3]
                if not check_angle(p2, p3, p4) or not check_angle(p3, p4, p1):
                    continue
            
            # Cost calculation
            g_new = current_node.g + 1
            
            # Heuristic selection based on mode
            if mode == "h1":
                h_new = heuristic_1(new_k)
            elif mode == "h2":
                h_new = heuristic_2(new_k, new_corners, bin_bool)
            else:
                h_new = 0.0 # Blind Search (Breadth-First)
                
            child_node = Node(corners=new_corners, g=g_new, h=h_new)
            heapq.heappush(open_set, child_node)

    search_time_elapsed = time.time() - search_start_time

    # --- Step 10: Ratio Calculation ---
    average_ratio = 0.0
    if bestScore >= 0.5:
        found = True

    if found:
        v1 = bestA4[0] - bestA4[1]
        v2 = bestA4[1] - bestA4[2]
        v3 = bestA4[2] - bestA4[3]
        v4 = bestA4[3] - bestA4[0]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        norm3 = np.linalg.norm(v3)
        norm4 = np.linalg.norm(v4)

        if norm1 > norm2:
            ratio1, ratio2 = norm1 / A4_HEIGHT_CM, norm3 / A4_HEIGHT_CM
            ratio3, ratio4 = norm2 / A4_WIDTH_CM, norm4 / A4_WIDTH_CM
        else:
            ratio1, ratio2 = norm2 / A4_HEIGHT_CM, norm4 / A4_HEIGHT_CM
            ratio3, ratio4 = norm1 / A4_WIDTH_CM, norm3 / A4_WIDTH_CM

        average_ratio = (ratio1 + ratio2 + ratio3 + ratio4) / 4.0

    return found, bestA4, bestScore, average_ratio, nodes_expanded, search_time_elapsed

# =============================================================================
# EMPIRICAL TESTING LAUNCHER
# =============================================================================
if __name__ == "__main__":
    
    image_path = '/Users/fathiahmad/UNM degree/Fundamentals of AI/testimage.png' 
    
    test_modes = ["h0", "h1", "h2"]
    mode_names = {
        "h0": "BREADTH-FIRST SEARCH (Blind)",
        "h1": "A* SEARCH WITH HEURISTIC 1",
        "h2": "A* SEARCH WITH HEURISTIC 2 (Smart)"
    }
    
    print("\n" + "="*50)
    print(" STARTING EMPIRICAL ANALYSIS")
    print("="*50)
    
    for mode in test_modes:
        print(f"\n---> RUNNING: {mode_names[mode]}")
        
        try:
            found, bestA4, score, ratio, nodes, time_taken = detect_a4_main(image_path, mode=mode)
            
            # Print individual report
            print(f"Status        : {'FOUND' if found else 'FAILED'}")
            print(f"Nodes Expanded: {nodes} nodes")
            print(f"Search Time   : {time_taken:.4f} seconds")
            print(f"Best Score    : {score:.3f}")
            if found:
                print(f"Pixel/CM Ratio: {ratio:.2f} px/cm")
                
        except Exception as e:
            print(f"Error occurred during {mode}: {e}")
            
    print("\n" + "="*50)
    print(" ANALYSIS COMPLETE. PLEASE COPY RESULTS TO REPORT.")
    print("="*50 + "\n")