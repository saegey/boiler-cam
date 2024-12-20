import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    validation_split=0.2,
    subset='training',
    image_size=(83, 61),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical',
    shuffle=True,
    seed=42,
    # labels='categorical'
)

# Extract y_train
y_train = []
for batch in train_ds:
    _, labels = batch
    y_train.extend(np.argmax(labels.numpy(), axis=1))  # Convert one-hot to integer labels

y_train = np.array(y_train)
class_names = train_ds.class_names

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/train',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(83, 61),
    batch_size=32,
    color_mode='grayscale',
    shuffle=False,
    label_mode='categorical',
    # labels='categorical',
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])

# Define the model
# Define the model
model = models.Sequential([
    layers.Input(shape=(83, 61, 1)),  # Use Input layer as recommended

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.shuffle(buffer_size=1000)

class_weights = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)
class_weights = dict(enumerate(class_weights))

lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 patience=3,
                                 verbose=1,
                                 factor=0.5,
                                 min_lr=1e-6)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# print(train_ds.class_names)

train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
model_checkpoint = ModelCheckpoint('digit_recognition_model.keras', monitor='val_loss', save_best_only=True)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,  # Adjust based on your needs
    class_weight=class_weights,
    callbacks=[lr_reduction, model_checkpoint]
    # callbacks=[early_stopping]
)

model.save('digit_recognition_model.keras')

model.summary()