
import os
import random
import shutil
import cv2

# Dataset source
raw_path = "data/raw"

# Destination for prepared data
output_path = "data/processed"

# Categories
eye_categories = ["Open", "Closed"]
yawn_categories = ["yawn", "no_yawn"]

# Split percentages
train_ratio = 0.70
val_ratio = 0.15

# Make results reproducible
random.seed(42)


def prepare_category(category, task):

    source_folder = os.path.join(raw_path, category)

    images = []

    # Read all image files
    for filename in os.listdir(source_folder):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(source_folder, filename)

            # Check whether image can be opened
            image = cv2.imread(image_path)

            if image is not None:
                images.append(filename)

    # Shuffle images
    random.shuffle(images)

    total = len(images)

    # Calculate split sizes
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Copy images into destination folders
    splits = {
        "train": train_images,
        "validation": val_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        destination = os.path.join(
            output_path, task, split_name, category
        )

        os.makedirs(destination, exist_ok=True)

        for filename in split_images:

            shutil.copy2(
                os.path.join(source_folder, filename),
                os.path.join(destination, filename)
            )

    print(
        category,
        "Total:", total,
        "Train:", len(train_images),
        "Validation:", len(val_images),
        "Test:", len(test_images)
    )


# Prepare eye dataset
print("\nPREPARING EYE DATASET")

for category in eye_categories:
    prepare_category(category, "eye")


# Prepare yawn dataset
print("\nPREPARING YAWN DATASET")

for category in yawn_categories:
    prepare_category(category, "yawn")


print("\nDataset preparation completed!")
