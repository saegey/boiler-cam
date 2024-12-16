import cv2
import numpy as np
import os
import sys
import json
import pytesseract
import re
from PIL import Image

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
rotation_matrix = cv2.getRotationMatrix2D(center, 1, 1.0)
rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (w_c, h_c), borderValue=(255, 255, 255))

# Convert to grayscale and apply Gaussian Blur + adaptive threshold
gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (11, 11), 0)


blurred_image_path = os.path.join(debug_folder, f"{epoch_timestamp}_blur_image.jpg")
cv2.imwrite(blurred_image_path, blurred_image)

grid_bw_image = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 41, 5
)

grid_bw_image_path = os.path.join(debug_folder, f"{epoch_timestamp}_grid_bw_image.jpg")
cv2.imwrite(grid_bw_image_path, grid_bw_image)

# Morphological opening
# kernel = np.ones((7, 7), np.uint8)
# morph_grid_bw_image = cv2.morphologyEx(grid_bw_image, cv2.MORPH_OPEN, kernel)

# Morphological closing
# kernel = np.ones((3, 3), np.uint8)
# closed_image = cv2.morphologyEx(morph_grid_bw_image, cv2.MORPH_CLOSE, kernel)

# Erosion
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
eroded_image = cv2.erode(grid_bw_image, kernel, iterations=4)

# Save the eroded image in the debug folder with epoch prefix
eroded_image_path = os.path.join(debug_folder, f"{epoch_timestamp}_eroded_image.jpg")
cv2.imwrite(eroded_image_path, eroded_image)
print(f"Eroded image saved: {eroded_image_path}")

# -----------------------
# Define Bounding Boxes
# -----------------------
# bounding_box_standby = (54, 29, 471, 101)  # (x1, y1, x2, y2)
bounding_box_standby = (54, 29, 912, 106)
bounding_box_system_temp_standby = (1025, 332, 1179, 402) # (x1, y1, x2, y2)

sensor_bounding_boxes = {
    "run_percentage": (293, 28, 407, 103),
    "flame": (964, 37, 1177, 109),
    "outlet temp": (1018, 182, 1186, 259),
    "inlet temp": (1025, 258, 1183, 328),
    "system temp": (735, 326, 899, 402),
    "setpoint temp": (1082, 330, 1245, 402),
}

# -----------------------
# OCR Logic with Tesseract
# -----------------------
# Configure Tesseract to return data including confidence
# You can specify the path to the Tesseract executable if it's not in PATH
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_crop(image, box, config=''):
    x1, y1, x2, y2 = box
    cropped = image[y1:y2, x1:x2]
    pil_image = Image.fromarray(cropped)
    # Use pytesseract to get detailed data including confidence
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=config)
    results = []
    num_boxes = len(data['level'])
    for i in range(num_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if text:
            results.append((data['left'][i], data['top'][i], data['width'][i], data['height'][i], text, conf))
    return results, cropped

def enhance_and_rerun_ocr(cropped_image, config=''):
    # Check if cropped_image is already single-channel or empty
    if cropped_image is None or cropped_image.size == 0:
        print("Warning: Cropped image is empty or invalid. Skipping enhancement.")
        return []

    # If it's a 3-channel image (BGR), convert to gray. If single-channel, assume it's already grayscale.
    if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    else:
        # Already single-channel or something unexpected
        gray = cropped_image

    # Proceed with thresholding and morphological operations
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    pil_image = Image.fromarray(cleaned)
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=config)
    results = []
    num_boxes = len(data['level'])
    for i in range(num_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if text:
            results.append((data['left'][i], data['top'][i], data['width'][i], data['height'][i], text, conf))
    return results

# Define Tesseract configuration
# --oem 3: Default OCR Engine Mode
# --psm 6: Assume a single uniform block of text
# You can adjust psm based on your specific use case
tess_config = '--oem 3 --psm 7'

# Check STANDBY
results, _ = ocr_crop(eroded_image, bounding_box_standby, config=tess_config + ' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-:.')

detected_text = None
for result in results:
    text = result[4].upper()
    confidence = result[5]
    detected_text = text.strip()

    if detected_text in ["STANDBY", "BLOCKED"]:
        print(f"Detected '{detected_text}' in the standby region.")

        # Process system temp for standby
        results_temp, cropped_system_temp = ocr_crop(eroded_image, bounding_box_system_temp_standby, config=tess_config + ' -c tessedit_char_whitelist=0123456789.')

        # If OCR fails, apply preprocessing and retry
        if not results_temp:
            print("No OCR results on initial system temp. Preprocessing...")
            results_temp = enhance_and_rerun_ocr(cropped_system_temp, config=tess_config + ' -c tessedit_char_whitelist=0123456789.')

        system_temp_data = {}
        if results_temp:
            # Choose the result with the highest confidence
            best_result = max(results_temp, key=lambda x: x[5])
            text_temp = best_result[4].strip()
            confidence_temp = best_result[5]
            system_temp_data = {
                "status": detected_text,
                "system temp": {
                    "value": text_temp,
                    "confidence": confidence_temp
                }
            }

        output_file = os.path.join(debug_folder, f"{epoch_timestamp}_sensor_data.json")
        with open(output_file, "w") as f:
            json.dump(system_temp_data, f, indent=4)
        print(f"System Sensor data written to '{output_file}'")

        sys.exit(0)

if detected_text != "STANDBY":
    print("No 'STANDBY' detected. Processing other sensor values...")
    sensor_data = {}
    sensor_data["status"] = detected_text if detected_text else "UNKNOWN"

    for sensor_name, box in sensor_bounding_boxes.items():
        # Define allowlist based on sensor type
        whitelist = '0123456789.'

        config = tess_config + f' -c tessedit_char_whitelist={whitelist}'

        results, cropped_sensor = ocr_crop(eroded_image, box, config=config)
        text_stripped = ""
        confidence = 0

        if results:
            # Choose the result with the highest confidence
            best_result = max(results, key=lambda x: x[5])
            text_stripped = best_result[4].strip()
            confidence = best_result[5]

        # Always enhance for specific keys
        if sensor_name in ["flame", "run_percentage"]:
            print(f"Always enhancing for '{sensor_name}'...")
            enhanced_results = enhance_and_rerun_ocr(cropped_sensor, config=config)
            if enhanced_results:
                best_enhanced = max(enhanced_results, key=lambda x: x[5])
                text_enhanced = best_enhanced[4].strip()
                confidence_enhanced = best_enhanced[5]
                print(f"After enhancement: '{sensor_name}': {text_enhanced} (Confidence: {confidence_enhanced})")
                text_stripped = text_enhanced
                confidence = confidence_enhanced
            else:
                print(f"No enhanced OCR result for '{sensor_name}'. Using initial OCR results.")

        # Special handling for "run_percentage" to mark as "invalid" if below 10%
        if sensor_name == "run_percentage":
            try:
                run_percentage = float(text_stripped.replace('%', ''))  # Remove '%' and convert to float
                if run_percentage < 10:
                    print(f"Run percentage below 10%: {run_percentage}. Marking as 'invalid'.")
                    text_stripped = "invalid"
                    confidence = 100  # Assign a high confidence for manual review
            except ValueError:
                print(f"Unable to parse run percentage: '{text_stripped}'. Marking as 'invalid'.")
                text_stripped = "invalid"
                confidence = 100  # Assign a high confidence for manual review

        # Special handling for "flame" to add a decimal point if needed
        if sensor_name == "flame" and '.' not in text_stripped and text_stripped != "invalid":
            print(f"Adding decimal point to flame value: '{text_stripped}'")
            try:
                # Add decimal one digit from the right
                if len(text_stripped) > 1:  # Ensure there's enough length to add a decimal
                    text_stripped = text_stripped[:-1] + '.' + text_stripped[-1]
                    print(f"Flame value adjusted to: '{text_stripped}'")
                else:
                    print(f"Flame value '{text_stripped}' is too short to adjust.")
            except Exception as e:
                print(f"Error adjusting flame value: {e}")

        sensor_data[sensor_name] = {
            "value": text_stripped,
            "confidence": confidence,
        }
        print(f"{sensor_name}: {text_stripped} (Confidence: {confidence})")

    output_file = os.path.join(debug_folder, f"{epoch_timestamp}_sensor_data.json")
    with open(output_file, "w") as f:
        json.dump(sensor_data, f, indent=4)
    print(f"Sensor data written to '{output_file}'")
