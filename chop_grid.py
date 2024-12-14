import cv2
import os
import glob

# Define the rows to include (0-indexed, skipping the 2nd and last rows)
include_rows = [0, 2, 3, 4]  # Rows to keep (excluding 1 and 5)

def process_image(image_path, left_buffer=-8, top_buffer=-4):
    """Processes a single image to add a grid with buffers."""
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Unable to load the image '{image_path}'.")
        return None

    # Get image dimensions
    height, width = image.shape

    # Define grid dimensions (22 columns)
    cols = 22
    cell_height = height // 6  # Divide height by total rows (6) to get the row height
    cell_width = width // cols

    # Create a color copy of the image for visualization
    grid_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Loop through the selected rows and all columns
    for row in include_rows:
        for col in range(cols):
            # Calculate cell boundaries with buffers
            x_start = max(0, col * cell_width - left_buffer)  # Ensure x_start is not negative
            x_end = x_start + cell_width
            y_start = max(0, row * cell_height - top_buffer)  # Ensure y_start is not negative
            y_end = y_start + cell_height

            # Draw the grid on the visualization image
            cv2.rectangle(grid_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 1)

    return grid_image

def main(input_dir):
    """Main function to process images in a directory and navigate through them."""
    # Get list of images in the directory
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if not image_paths:
        print(f"No images found in the directory: {input_dir}")
        return

    print("Use arrow keys to navigate images. Press 'q' to quit.")

    index = 0
    while True:
        # Load and process the current image
        image_path = image_paths[index]
        grid_image = process_image(image_path)

        # Display the processed image
        if grid_image is not None:
            filename = os.path.basename(image_path)
            cv2.imshow(f"Grid Visualization - {filename}", grid_image)

        # Wait for key input
        key = cv2.waitKey(0)

        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord('w'):  # Left arrow key (←)
            index = (index - 1) % len(image_paths)  # Navigate left
        elif key == ord('e'):  # Right arrow key (→)
            index = (index + 1) % len(image_paths)  # Navigate right

        # Close current window before displaying the next
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the directory containing images
    input_directory = "debug"  # Replace with your directory path
    main(input_directory)
