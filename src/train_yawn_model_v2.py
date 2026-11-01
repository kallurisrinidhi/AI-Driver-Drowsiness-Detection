
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Dataset paths
train_path = "data/processed/yawn/train"
val_path = "data/processed/yawn/validation"

# Load datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(64, 64),
    batch_size=32,
    label_mode="binary",
    shuffle=True,
    seed=42
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=(64, 64),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

print("Classes:", train_dataset.class_names)

# Improve data loading
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

val_dataset = val_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Stop when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Save the new model separately
checkpoint = ModelCheckpoint(
    "models/yawn_classifier_v2.keras",
    monitor="val_loss",
    save_best_only=True
)

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=25,
    callbacks=[early_stopping, checkpoint]
)

print("New yawn model training completed!")
print("Saved: models/yawn_classifier_v2.keras")
