import tensorflow as tf
from tensorflow.keras import layers, models

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(61, 83),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(61, 83),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical'
)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(61, 83, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Add dropout for regularization
    layers.Dense(10, activation='softmax')  # Output layer for 25 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(train_ds.class_names)

train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # Adjust based on your needs
)

model.save('digit_recognition_model.h5')
