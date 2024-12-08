import os
import re
import sys
import cv2
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

# Set the directory containing the results
if len(sys.argv) < 2:
    print("Usage: python qa_viewer.py <debug_folder_path>")
    sys.exit(1)

debug_folder = sys.argv[1]
if not os.path.isdir(debug_folder):
    print(f"Error: '{debug_folder}' is not a directory.")
    sys.exit(1)

# Find all eroded images in the debug folder
eroded_images = glob.glob(os.path.join(debug_folder, "*_eroded_image.jpg"))

# Extract epochs from filenames and sort
def extract_epoch(filename):
    # Expecting something like "<epoch>_eroded_image.jpg"
    # We'll use a regex to find the epoch number
    base = os.path.basename(filename)
    match = re.match(r"(\d+)_eroded_image\.jpg", base)
    if match:
        return int(match.group(1))
    return None

eroded_images = [img for img in eroded_images if extract_epoch(img) is not None]
eroded_images.sort(key=lambda x: extract_epoch(x))

if not eroded_images:
    print("No eroded images found in the specified directory.")
    sys.exit(0)

# For each eroded image, find a corresponding JSON file (sensor_data or standby)
# Priority: if "system_temp_standby" exists, use it, otherwise use "sensor_data"
results_pairs = []
for img_path in eroded_images:
    epoch = extract_epoch(img_path)
    if epoch is None:
        continue

    # Look for JSON files with this epoch
    json_sensor = os.path.join(debug_folder, f"{epoch}_sensor_data.json")
    json_standby = os.path.join(debug_folder, f"{epoch}_system_temp_standby.json")

    if os.path.exists(json_standby):
        json_path = json_standby
    elif os.path.exists(json_sensor):
        json_path = json_sensor
    else:
        json_path = None

    # Add to list even if json not found, we can handle that gracefully
    results_pairs.append((img_path, json_path))

current_index = 0

# Setup matplotlib figure
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.canvas.manager.set_window_title("QA Viewer - Press LEFT/RIGHT to Navigate")

def display_current():
    axs[0].clear()
    axs[1].clear()
    axs[0].set_title("Eroded Image")
    axs[1].set_title("OCR JSON Data")

    img_path, json_path = results_pairs[current_index]

    # Load image and display
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[0].imshow(image_rgb)
        axs[0].axis('off')
    else:
        axs[0].text(0.5, 0.5, "Image not found or unable to load", ha='center', va='center')
        axs[0].axis('off')

    # Load JSON and display
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        json_str = json.dumps(data, indent=4)
        axs[1].axis('off')
        axs[1].text(0.01, 0.99, json_str, family='monospace', fontsize=10, va='top', ha='left')
    else:
        axs[1].axis('off')
        axs[1].text(0.5, 0.5, "No JSON file found", ha='center', va='center')

    # Set a global title showing current index and total
    total = len(results_pairs)
    fig.suptitle(f"Record {current_index+1}/{total} (Epoch: {extract_epoch(img_path)})", fontsize=14)
    plt.draw()

def on_key(event: KeyEvent):
    global current_index
    if event.key == 'right':
        if current_index < len(results_pairs) - 1:
            current_index += 1
            display_current()
    elif event.key == 'left':
        if current_index > 0:
            current_index -= 1
            display_current()

fig.canvas.mpl_connect('key_press_event', on_key)

# Display the first one
display_current()
plt.tight_layout()
plt.show()
