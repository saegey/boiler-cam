import os
import tensorflow as tf
import numpy as np
import shutil
from PIL import Image
import matplotlib.pyplot as plt

def classify_with_model(model_path, input_folder, output_folder, image_size=(61, 83), confidence_threshold=0.9):
    """
    Classifies images using a trained model and organizes them into folders by predicted class,
    only if confidence exceeds the given threshold.

    Args:
        model_path (str): Path to the trained model file (e.g., .h5).
        input_folder (str): Path to the folder containing leftover images.
        output_folder (str): Path to the folder for classified images.
        image_size (tuple): Input size of the model (e.g., (61, 83)).
        confidence_threshold (float): Minimum confidence score to classify an image.
    """
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    confidences = []  # To track confidence scores

    # Process each image in the input folder
    for image_file in sorted(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, image_file)

        # Skip non-image files
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load and preprocess the image
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        image = image.resize(image_size)  # Resize to model input size
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=(0, -1))  # Add batch and channel dimensions

        # Predict using the model
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        confidences.append(confidence)  # Track confidence score

        # Only classify images with confidence above the threshold
        if confidence >= confidence_threshold:
            # Get class label (e.g., 0-9, A-Z)
            class_label = f"{predicted_class}"

            # Create folder for the predicted class
            class_folder = os.path.join(output_folder, class_label)
            os.makedirs(class_folder, exist_ok=True)

            # Move the image to the appropriate folder
            destination_path = os.path.join(class_folder, image_file)
            shutil.move(image_path, destination_path)

            print(f"Classified '{image_file}' as '{class_label}' with confidence {confidence:.2f}.")
        else:
            print(f"Skipped '{image_file}' due to low confidence ({confidence:.2f}).")

    # Plot confidence score distribution
    plt.hist(confidences, bins=20, range=(0, 1), edgecolor='black')
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    print("Classification completed. Manually QA the results in the output folder.")

if __name__ == "__main__":
    # Paths
    model_path = "digit_recognition_model.keras"  # Path to your trained model
    input_folder = "output_characters"  # Folder with leftover images
    output_folder = "classified_images"  # Folder to store classified images

    # Classify leftover images
    classify_with_model(model_path, input_folder, output_folder, confidence_threshold=0.80)