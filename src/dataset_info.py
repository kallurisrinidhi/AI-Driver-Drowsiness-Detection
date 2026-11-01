
import os

# Location of our dataset
dataset_path = "data/raw"

# Dataset categories
categories = ["Open", "Closed", "yawn", "no_yawn"]

total_images = 0

for category in categories:

    folder_path = os.path.join(dataset_path, category)

    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = len(images)

    print(category, ":", count, "images")

    total_images += count

print("--------------------------")
print("Total images:", total_images)
