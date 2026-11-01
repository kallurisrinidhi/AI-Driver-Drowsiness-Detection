
import tensorflow as tf
import os

# 1. Dataset paths

train_path = "data/processed/yawn/train"
validation_path = "data/processed/yawn/validation"
test_path = "data/processed/yawn/test"

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32

# 2. Load training dataset

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

# 3. Load validation dataset

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# 4. Load test dataset

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Yawn classes:", train_dataset.class_names)

# 5. Improve data loading

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# 6. Build CNN model

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(64, 64, 3)),

    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128, activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1, activation="sigmoid"
    )
])

# 7. Compile model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 8. Display model architecture

model.summary()

# 9. Early stopping

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# 10. Train model

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15,
    callbacks=[early_stopping]
)

# 11. Evaluate model

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\nYAWN MODEL EVALUATION")

print("Test Loss:", test_loss)

print("Test Accuracy:", test_accuracy)

# 12. Save model

os.makedirs("models", exist_ok=True)

model.save("models/yawn_classifier.keras")

print("\nYawn model saved successfully!")
