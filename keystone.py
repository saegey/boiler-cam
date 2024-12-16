import cv2
import numpy as np
import json

# Initialize global variables
points = []  # To store points selected by the user
temp_image = None  # Temporary image for visualization

def select_points(event, x, y, flags, param):
    """Mouse callback function to capture points."""
    global points, temp_image
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point and display it
        points.append((x, y))
        cv2.circle(temp_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Corners", temp_image)

def keystone_correction_gui(image_path, save_points_path="points.json"):
    """Interactive GUI for keystone correction and save selected points."""
    global points, temp_image

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image. Check the path!")
        return

    # Make a copy of the image for point selection
    temp_image = image.copy()

    # Create a window and set a mouse callback
    cv2.imshow("Select Corners", temp_image)
    cv2.setMouseCallback("Select Corners", select_points)

    print("Please select the four corners of the trapezoid (in order: top-left, top-right, bottom-left, bottom-right).")

    # Wait for user to select exactly 4 points
    while len(points) < 4:
        cv2.waitKey(1)

    # Close the selection window
    cv2.destroyAllWindows()

    # Save points to a file for reuse
    with open(save_points_path, "w") as file:
        json.dump(points, file)
    print(f"Selected points saved to {save_points_path}.")

    # Perform the correction using the selected points
    correct_keystone(image, points)

def correct_keystone(image, source_points, save_output_path="corrected_image.jpg"):
    """Perform keystone correction using given points."""
    # Define destination points (rectangle with the same size as the original image)
    height, width = image.shape[:2]
    destination_points = np.float32([
        [0, 0],                     # Top-left corner
        [width, 0],                 # Top-right corner
        [0, height],                # Bottom-left corner
        [width, height]             # Bottom-right corner
    ])

    # Convert the selected points to a NumPy array
    source_points = np.float32(source_points)

    # Compute the transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(source_points, destination_points)

    # Apply the warp perspective
    corrected_image = cv2.warpPerspective(image, transformation_matrix, (width, height))

    # Show the corrected image
    cv2.imshow("Corrected Image", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the corrected image
    cv2.imwrite(save_output_path, corrected_image)
    print(f"Corrected image saved to {save_output_path}.")

# Replace this with your actual image path
image_path = "debug/1733596831_eroded_image.jpg"
keystone_correction_gui(image_path)
