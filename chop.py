import cv2
import os
import sys
import re

# Define the specific cells (row, columns) to export
include_cells = {
    0: [4, 5, 16, 17, 19],
    2: [17, 18, 19],
    3: [17, 18, 19],
    4: [18, 19, 20, 14, 13, 12],
}

def extract_epoch_from_filename(filename):
    """Extract the epoch timestamp from the filename."""
    match = re.search(r"(\d+)_keystone_corrected", filename)
    if match:
        return match.group(1)
    return "unknown"

def process_image(image_path, output_folder, debug_folder, left_buffer=-8, top_buffer=-4):
    """Processes a single image to export specified grid sections."""
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load the image '{image_path}'.")
        return

    # Extract epoch from the filename
    filename = os.path.basename(image_path)
    epoch = extract_epoch_from_filename(filename)

    # Get image dimensions
    height, width = image.shape

    # Define grid dimensions (22 columns)
    cols = 22
    cell_height = height // 6  # Divide height by total rows (6) to get the row height
    cell_width = width // cols

    # Create a color copy of the image for visualization
    grid_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    # Loop through the defined cells
    for row, cols in include_cells.items():
        for col in cols:
            # Calculate cell boundaries with buffers
            x_start = max(0, col * cell_width - left_buffer)  # Ensure x_start is not negative
            x_end = min(width, x_start + cell_width)          # Ensure x_end doesn't exceed image width
            y_start = max(0, row * cell_height - top_buffer)  # Ensure y_start is not negative
            y_end = min(height, y_start + cell_height)        # Ensure y_end doesn't exceed image height

            # Draw the grid on the visualization image
            cv2.rectangle(grid_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

            # Extract the cell with buffers
            character = image[y_start:y_end, x_start:x_end]

            # Save the character image with epoch in the filename
            output_path = os.path.join(output_folder, f"{epoch}_char_{row}_{col}.jpg")
            cv2.imwrite(output_path, character)
            print(f"Exported: {output_path}")

    # Save and display the grid visualization
    grid_image_path = os.path.join(debug_folder, f"{epoch}_grid_visualization.jpg")
    cv2.imwrite(grid_image_path, grid_image)
    print(f"Grid visualization saved: {grid_image_path}")

def main():
    """Main function to process an image passed as an argument."""
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    # Specify the output folder for exported images
    output_folder = "output_characters"
    debug_folder = "debug"

    # Process the image
    process_image(image_path, output_folder, debug_folder)

if __name__ == "__main__":
    main()