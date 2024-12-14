import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the model
model = tf.keras.models.load_model('digit_recognition_model.h5')

# Load test data
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "test",
    image_size=(61, 83),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical'
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Visualize predictions
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Plot some test images with their predictions
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f"True: {labels[i]}, Pred: {predicted_classes[i]}")
        plt.axis("off")
    plt.show()
