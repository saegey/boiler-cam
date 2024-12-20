import cv2
import numpy as np
import os
import sys
import json
import re

if len(sys.argv) < 2:
    print("Usage: python final_with_epoch_and_eroded.py <image_filename>")
    sys.exit(1)

image_path = sys.argv[1]

# Check if the file exists
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
    sys.exit(1)

# Extract epoch from filename (assuming format: image-<epoch>...)
filename = os.path.basename(image_path)
match = re.search(r"image-(\d+)", filename)
if not match:
    print("Error: Could not extract epoch timestamp from the filename.")
    sys.exit(1)
epoch_timestamp = match.group(1)

# Create debug folder if it doesn't exist
debug_folder = os.path.join(os.getcwd(), "debug")
os.makedirs(debug_folder, exist_ok=True)

# ------------------
# Preprocessing Steps
# ------------------
# Load the original image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image '{image_path}'.")
    sys.exit(1)

# Initial crop: equivalent to `-crop 1350x500+450+370`
x, y, w, h = 450, 370, 1350, 500
if (x + w) > image.shape[1] or (y + h) > image.shape[0]:
    print("Error: Initial crop dimensions out of range.")
    sys.exit(1)
cropped_image = image[y:y+h, x:x+w]

# Rotate the image by 1 degree
(h_c, w_c) = cropped_image.shape[:2]
center = (w_c // 2, h_c // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 1.3, 1.0)
rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (w_c, h_c), borderValue=(255, 255, 255))

# Convert to grayscale and apply Gaussian Blur + adaptive threshold
gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)

# blurred_image_path = os.path.join(debug_folder, f"{epoch_timestamp}_blur_image.jpg")
# cv2.imwrite(blurred_image_path, blurred_image)

grid_bw_image = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 41, 5
)

# grid_bw_image_path = os.path.join(debug_folder, f"{epoch_timestamp}_grid_bw_image.jpg")
# cv2.imwrite(grid_bw_image_path, grid_bw_image)

# Erosion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
eroded_image = cv2.erode(grid_bw_image, kernel, iterations=4)

# Save the eroded image in the debug folder with epoch prefix
# eroded_image_path = os.path.join(debug_folder, f"{epoch_timestamp}_eroded_image.jpg")
# cv2.imwrite(eroded_image_path, eroded_image)
# print(f"Eroded image saved: {eroded_image_path}")

# ------------------
# Keystone Correction
# ------------------
# Load points from points.json
points_file = "points.json"
if not os.path.exists(points_file):
    print(f"Error: '{points_file}' not found. Cannot perform keystone correction.")
    sys.exit(1)

with open(points_file, "r") as file:
    points = json.load(file)

# Convert points to NumPy array
source_points = np.float32(points)

# Define destination points for rectangle with same size as eroded image
height, width = eroded_image.shape[:2]
destination_points = np.float32([
    [0, 0],                # Top-left corner
    [width, 0],            # Top-right corner
    [0, height],           # Bottom-left corner
    [width, height]        # Bottom-right corner
])

# Compute transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(source_points, destination_points)

# Apply the keystone correction
keystone_corrected_image = cv2.warpPerspective(eroded_image, transformation_matrix, (width, height))

# Save the keystone-corrected image
keystone_corrected_image_path = os.path.join(debug_folder, f"{epoch_timestamp}_keystone_corrected_image.jpg")
cv2.imwrite(keystone_corrected_image_path, keystone_corrected_image)
print(f"Keystone-corrected image saved: {keystone_corrected_image_path}")
