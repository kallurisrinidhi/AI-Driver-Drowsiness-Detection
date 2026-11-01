
import tensorflow as tf

# Dataset folders
train_path = "data/processed/eye/train"
validation_path = "data/processed/eye/validation"
test_path = "data/processed/eye/test"

# Image settings
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32

# Load training images
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=True,
    seed=42
)

# Load validation images
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

# Load testing images
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    color_mode="rgb",
    shuffle=False
)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

test_dataset = test_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Display information
print("Class names:", train_dataset.class_names
      if hasattr(train_dataset, "class_names") else "Loaded")

print("Training dataset loaded successfully!")
print("Validation dataset loaded successfully!")
print("Testing dataset loaded successfully!")

# Check one batch
for images, labels in train_dataset.take(1):

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Minimum pixel value:", tf.reduce_min(images).numpy())
    print("Maximum pixel value:", tf.reduce_max(images).numpy())
