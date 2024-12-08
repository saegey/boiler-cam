import cv2
import numpy as np
import os
import sys

# Ensure there is an input image argument
if len(sys.argv) < 2:
    print("Usage: python script.py <image_filename>")
    sys.exit(1)

# 1. Capture the image file path from the command-line argument
input_image_path = sys.argv[1]

# Check if the image file exists
if not os.path.exists(input_image_path):
    print(f"Error: File '{input_image_path}' not found.")
    sys.exit(1)

# 2. Load the captured image
image = cv2.imread(input_image_path)
if image is None:
    print(f"Error: Unable to read the image file '{input_image_path}'.")
    sys.exit(1)

# Get the image dimensions
height, width = image.shape[:2]

# Define cropping function
def crop_image(event, x, y, flags, param):
    global cropping, x_start, y_start, x_end, y_end

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button pressed
        cropping = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse is moving
        if cropping:
            x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:  # Left mouse button released
        cropping = False
        x_end, y_end = x, y

        # Ensure bounding box coordinates are within image bounds
        x1, x2 = sorted([x_start, x_end])  # Sort x coordinates
        y1, y2 = sorted([y_start, y_end])  # Sort y coordinates

        x1 = max(0, min(x1, width - 1))  # Clamp x1 to image bounds
        x2 = max(0, min(x2, width - 1))  # Clamp x2 to image bounds
        y1 = max(0, min(y1, height - 1))  # Clamp y1 to image bounds
        y2 = max(0, min(y2, height - 1))  # Clamp y2 to image bounds

        # Crop the selected region
        cropped = image[y1:y2, x1:x2]

        # Display the cropped section
        cv2.imshow("Cropped", cropped)

        # Optionally save the cropped section
        cv2.imwrite("cropped_section.jpg", cropped)

        # Print bounding box for debugging
        print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

# Initialize variables
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

# Display the image and set up the mouse callback
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", crop_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
