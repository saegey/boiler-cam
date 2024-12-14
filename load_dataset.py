import tensorflow as tf

train_dir = "data/train"
val_dir = "data/validation"

img_height = 83  # or whatever size you choose
img_width = 61

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(83, 61),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(83, 61),
    batch_size=32
)
