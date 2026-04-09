import os
import csv
import cv2
import numpy as np

# Import the A* Engine function from your main detection file
from A4detection import detect_a4_main

# Set target dimensions manually to match the A* engine's preprocessing
TARGET_HEIGHT = 2800
TARGET_WIDTH = 2100

def draw_a4_rectangle(image, corners, color=(0, 0, 255), thickness=4):
    """
    Function to draw a red bounding box (rectangle) on the image.
    """
    num_corners = corners.shape[0]
    for i in range(num_corners):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % num_corners]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    subfolder_name = "Dataset"
    input_folder = os.path.join(current_dir, subfolder_name)
    output_folder = os.path.join(input_folder, "results")
    
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(input_folder, "a4_detection_results.csv")
    valid_extensions = {".jpg", ".jpeg", ".png"}

    print("========================================")
    print(" STARTING BATCH PROCESSING")
    print("========================================")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", "Found", "BestScore", "AvgRatio",
            "Corner1_x", "Corner1_y",
            "Corner2_x", "Corner2_y",
            "Corner3_x", "Corner3_y",
            "Corner4_x", "Corner4_y",
        ])

        for filename in os.listdir(input_folder):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                continue

            img_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}...")

            # Call the A* function. 
            # NOTE: We expect 6 outputs, using "h2" (Smart Heuristic)
            found, bestA4, bestScore, avg_ratio, nodes, time_taken = detect_a4_main(img_path, mode="h2")

            if found:
                c1x, c1y = bestA4[0]
                c2x, c2y = bestA4[1]
                c3x, c3y = bestA4[2]
                c4x, c4y = bestA4[3]
                
                row_data = [
                    filename, found, f"{bestScore:.3f}", f"{avg_ratio:.3f}",
                    f"{c1x:.1f}", f"{c1y:.1f}",
                    f"{c2x:.1f}", f"{c2y:.1f}",
                    f"{c3x:.1f}", f"{c3y:.1f}",
                    f"{c4x:.1f}", f"{c4y:.1f}",
                ]
            else:
                row_data = [filename, found, "", "", "", "", "", "", "", "", "", ""]

            writer.writerow(row_data)

            # DRAW THE BOUNDING BOX ON THE OUTPUT IMAGE
            if found:
                image_for_overlay = cv2.imread(img_path)
                if image_for_overlay is not None:
                    # Since our A* algorithm evaluates on a 2100x2800 canvas,
                    # we must resize the original image first before drawing.
                    # Also, if the image is landscape, rotate it to match preprocessing.
                    
                    h, w = image_for_overlay.shape[:2]
                    if h < w:
                        image_for_overlay = np.rot90(image_for_overlay, k=3)
                        
                    overlay_img = cv2.resize(image_for_overlay, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

                    # Draw the lines! (Thickness set to 4 for better visibility)
                    draw_a4_rectangle(overlay_img, bestA4, color=(0, 0, 255), thickness=4)

                    out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_overlay.jpg")
                    cv2.imwrite(out_path, overlay_img)

            print("Done.")
            
    print("\nBatch Processing Complete! Please check the 'results' folder and the CSV file.")