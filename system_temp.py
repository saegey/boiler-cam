import easyocr
import sys
import os
import cv2
import json

# Ensure there is an input image argument
if len(sys.argv) < 2:
    print("Usage: python script.py <image_filename>")
    sys.exit(1)

# Path to the input image file
image_path = sys.argv[1]

# Check if the file exists
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
    sys.exit(1)

# Bounding box coordinates
bounding_box_standby = (54, 29, 471, 101)  # (x1, y1, x2, y2)
bounding_box_system_temp_standby = (1071, 330, 1191, 399)  # (x1, y1, x2, y2)
sensor_bounding_boxes = {
    "run %": (293, 28, 407, 103),
    "flame": (964, 37, 1177, 109),
    "outlet temp": (1018, 182, 1186, 259),
    "inlet temp": (1025, 258, 1183, 328),
    "system temp": (735, 326, 899, 402),
    "setpoint temp": (1082, 330, 1245, 402),
}

# Read the image using OpenCV
image = cv2.imread(image_path)

# Check if the image was successfully loaded
if image is None:
    print(f"Error: Unable to load image '{image_path}'.")
    sys.exit(1)

# Crop the image for the "STANDBY" check
x1, y1, x2, y2 = bounding_box_standby
cropped_image_standby = image[y1:y2, x1:x2]

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Use 'gpu=True' if you have a compatible GPU

# Perform OCR on the cropped image
results = reader.readtext(cropped_image_standby, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Check if the detected text is "STANDBY"
detected_text = None
for (bbox, text, confidence) in results:
    detected_text = text.strip().upper()
    print(f"Detected Text: {detected_text}, Confidence: {confidence}")
    if detected_text == "STANDBY":
        print("The text is 'STANDBY'. Extracting 'system temp'...")
        # Crop and process system temp for STANDBY
        x1, y1, x2, y2 = bounding_box_system_temp_standby
        cropped_system_temp_image = image[y1:y2, x1:x2]

        # Save the cropped system temp image for debugging
        debug_image_path = "system_temp_debug.jpg"
        cv2.imwrite(debug_image_path, cropped_system_temp_image)
        print(f"System temp cropped image saved as '{debug_image_path}' for debugging.")

        # Perform OCR on the cropped system temp image
        results_temp = reader.readtext(cropped_system_temp_image, allowlist='0123456789.')

        # If OCR fails, apply preprocessing
        if not results_temp:
            print("OCR failed on initial system temp image. Applying preprocessing...")
            gray = cv2.cvtColor(cropped_system_temp_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Save the preprocessed image for debugging
            preprocessed_path = "preprocessed_system_temp.jpg"
            cv2.imwrite(preprocessed_path, cleaned)
            print(f"Preprocessed system temp image saved as '{preprocessed_path}'.")

            # Retry OCR on preprocessed image
            results_temp = reader.readtext(cleaned, allowlist='0123456789.')

        print(results_temp)

        system_temp_data = {}
        for (bbox_temp, text_temp, confidence_temp) in results_temp:
            system_temp_data = {
                "system temp": {
                    "value": text_temp.strip(),
                    "confidence": confidence_temp
                }
            }
            break  # Only take the first detected value

        # Write to JSON
        output_file = "system_temp_standby.json"
        with open(output_file, "w") as f:
            json.dump(system_temp_data, f, indent=4)
        print(f"System temp written to '{output_file}'")
        sys.exit(0)

print("The text is not 'STANDBY'. Processing other sensor values...")

# Process sensor values
sensor_data = {}
for sensor_name, (x1, y1, x2, y2) in sensor_bounding_boxes.items():
    # Crop the image for the sensor
    cropped_sensor_image = image[y1:y2, x1:x2]

    # Perform OCR on the cropped sensor image
    results = reader.readtext(cropped_sensor_image, allowlist='0123456789.%')

    # Extract the text and confidence
    for (bbox, text, confidence) in results:
        sensor_data[sensor_name] = {
            "value": text.strip(),
            "confidence": confidence,
        }
        print(f"{sensor_name}: {text.strip()} (Confidence: {confidence})")
        break  # Only take the first detected value per crop

# Write sensor data to a JSON file
output_file = "sensor_data.json"
with open(output_file, "w") as f:
    json.dump(sensor_data, f, indent=4)

print(f"Sensor data written to '{output_file}'")
