import os
import shutil
import easyocr
from PIL import Image

def process_images(input_folder, identified_folder, confidence_threshold=98):
    """
    Processes images with EasyOCR and organizes them into character-named folders.
    For low-confidence results, prompts the user to manually input the character.

    Args:
        input_folder (str): Path to the folder containing chopped images.
        identified_folder (str): Path to the folder for identified high-confidence images.
        confidence_threshold (float): Minimum confidence score to consider a result identified.
    """
    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'], gpu=True)

    # Create identified folder if it doesn't exist
    os.makedirs(identified_folder, exist_ok=True)

    # Process each image in the input folder
    for image_file in sorted(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, image_file)

        # Skip non-image files
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Run OCR on the image
        results = reader.readtext(image_path, detail=1)
        recognized_text = None
        confidence = 0

        # Check if OCR found any results
        if results:
            # Get the best result (highest confidence score)
            best_result = max(results, key=lambda x: x[2])
            recognized_text = best_result[1]
            confidence = best_result[2] * 100

        # Handle high-confidence results
        if recognized_text and confidence >= confidence_threshold:
            character = ''.join(e for e in recognized_text if e.isalnum())  # Keep alphanumeric characters only

            # Create a folder for the character if it doesn't exist
            character_folder = os.path.join(identified_folder, character)
            os.makedirs(character_folder, exist_ok=True)

            # Move the image to the character's folder
            destination_path = os.path.join(character_folder, image_file)
            shutil.move(image_path, destination_path)
            print(f"Saved '{image_file}' as '{character}/{image_file}'.")

if __name__ == "__main__":
    # Define input and output folders
    input_folder = "output_characters_with_buffers"  # Folder containing chopped images
    identified_folder = "identified"  # Folder to store identified high-confidence images

    # Process images
    process_images(input_folder, identified_folder)
