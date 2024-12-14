import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report

model = tf.keras.models.load_model('digit_recognition_model.h5')

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "test",
    image_size=(61, 83),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical'
)

# Collect true and predicted labels
for images, labels in test_ds:
    predictions = model.predict(images)

    # Convert predictions to class indices
    predicted_classes = np.argmax(predictions, axis=1)

    # Append labels and predictions
    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted_classes)

# Convert to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Ensure labels are integers
if len(true_labels.shape) > 1 and true_labels.shape[1] > 1:  # If one-hot encoded
    true_labels = np.argmax(true_labels, axis=1)
if len(predicted_labels.shape) > 1 and predicted_labels.shape[1] > 1:  # If one-hot encoded
    predicted_labels = np.argmax(predicted_labels, axis=1)

# Generate the classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=test_ds.class_names))
