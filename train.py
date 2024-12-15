import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])

# Define the model
# Define the model
model = models.Sequential([
    layers.Input(shape=(61, 83, 1)),
    layers.Rescaling(1./255),
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# model = models.Sequential([
#     layers.Input(shape=(61, 83, 1)),
#     layers.Rescaling(1./255),
#     data_augmentation,
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Dropout(0.2),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Dropout(0.2),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(10, activation='softmax')
# ])


train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.shuffle(buffer_size=1000)

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=0.001,
#     decay_steps=1000,
#     decay_rate=0.96,
#     staircase=True
# )

normalized_weights = {
    0: 0.1007,
    1: 0.0266,
    2: 0.0601,
    3: 0.0570,
    4: 0.1370,
    5: 0.1704,
    6: 0.1085,
    7: 0.1364,
    8: 0.1011,
    9: 0.1021
}


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-6, verbose=1)

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

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,  # Adjust based on your needs
    class_weight=normalized_weights,
    callbacks=[reduce_lr]
    # callbacks=[early_stopping]
)

model.save('digit_recognition_model.keras')

model.summary()