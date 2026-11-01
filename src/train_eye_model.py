
import tensorflow as tf
import os

# ---------------------------------
# 1. Dataset paths
# ---------------------------------

train_path = "data/processed/eye/train"
validation_path = "data/processed/eye/validation"
test_path = "data/processed/eye/test"

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32

# ---------------------------------
# 2. Load datasets
# ---------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Class names:", train_dataset.class_names)

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# ---------------------------------
# 3. Build CNN model
# ---------------------------------

model = tf.keras.Sequential([

    # Input image
    tf.keras.layers.Input(shape=(64, 64, 3)),

    # Normalize pixel values
    tf.keras.layers.Rescaling(1./255),

    # First convolution block
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Third convolution block
    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convert features into one dimension
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(128, activation="relu"),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.5),

    # Binary classification output
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# ---------------------------------
# 4. Compile the model
# ---------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()

# ---------------------------------
# 5. Stop when validation stops improving
# ---------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# ---------------------------------
# 6. Train the model
# ---------------------------------

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15,
    callbacks=[early_stopping]
)

# ---------------------------------
# 7. Evaluate the model
# ---------------------------------

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\nMODEL EVALUATION")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ---------------------------------
# 8. Save the trained model
# ---------------------------------

os.makedirs("models", exist_ok=True)

model.save("models/eye_classifier.keras")

print("\nModel saved successfully!")
