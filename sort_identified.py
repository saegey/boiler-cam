import os
import shutil

def sort_identified_images(identified_folder):
    """
    Sorts identified images into subfolders based on the character prefix.

    Args:
        identified_folder (str): Path to the folder containing identified images.
    """
    # Ensure the identified folder exists
    if not os.path.exists(identified_folder):
        print(f"Error: Identified folder '{identified_folder}' does not exist.")
        return

    # Iterate through all files in the identified folder
    for image_file in os.listdir(identified_folder):
        image_path = os.path.join(identified_folder, image_file)

        # Skip non-image files
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Extract the character prefix from the filename (text before the first '_')
        prefix = image_file.split('_')[0]

        # Create a subfolder for the character if it doesn't exist
        subfolder_path = os.path.join(identified_folder, prefix)
        os.makedirs(subfolder_path, exist_ok=True)

        # Move the file into the subfolder
        destination_path = os.path.join(subfolder_path, image_file)
        shutil.move(image_path, destination_path)
        print(f"Moved '{image_file}' to '{subfolder_path}'.")

if __name__ == "__main__":
    # Specify the identified folder path
    identified_folder = "identified"  # Replace with your actual folder path

    # Sort the images into subfolders
    sort_identified_images(identified_folder)
