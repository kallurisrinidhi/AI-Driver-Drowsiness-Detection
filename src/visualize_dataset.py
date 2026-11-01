
import os
import cv2
import matplotlib.pyplot as plt
import random

# Dataset location
dataset_path = "data/raw"

# Four categories
categories = ["Open", "Closed", "yawn", "no_yawn"]

# Create a figure with 4 rows and 4 columns
plt.figure(figsize=(12, 10))

for row, category in enumerate(categories):

    folder_path = os.path.join(dataset_path, category)

    # Get image filenames
    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Select 4 random images
    selected_images = random.sample(images, min(4, len(images)))

    for col, image_name in enumerate(selected_images):

        image_path = os.path.join(folder_path, image_name)

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            print("Could not read:", image_path)
            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display image
        plt.subplot(4, 4, row * 4 + col + 1)
        plt.imshow(image)
        plt.title(category)
        plt.axis("off")

plt.tight_layout()
plt.show()
