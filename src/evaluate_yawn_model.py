
import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Load trained yawn model
model = tf.keras.models.load_model(
    "models/yawn_classifier.keras"
)

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "data/processed/yawn/test",
    image_size=(64, 64),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

# Get class names
class_names = test_dataset.class_names

print("Classes:", class_names)

# Evaluate model
loss, accuracy = model.evaluate(test_dataset)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)

# Get actual labels
y_true = np.concatenate([
    labels.numpy().ravel()
    for images, labels in test_dataset
])

# Get predictions
predictions = model.predict(test_dataset)

y_pred = (predictions.ravel() >= 0.5).astype(int)

# Classification report
print("\nClassification Report")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# Confusion matrix
print("\nConfusion Matrix")

print(confusion_matrix(y_true, y_pred))
