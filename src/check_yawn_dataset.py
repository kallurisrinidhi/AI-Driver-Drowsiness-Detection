
import os
import cv2
import matplotlib.pyplot as plt

dataset_path = "data/processed/yawn"

splits = ["train", "validation", "test"]
classes = ["no_yawn", "yawn"]

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for row, split in enumerate(splits):

    for col, class_name in enumerate(classes):

        folder = os.path.join(
            dataset_path,
            split,
            class_name
        )

        images = [
            file for file in os.listdir(folder)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        print(
            split, class_name,
            ":", len(images), "images"
        )

        if images:

            image_path = os.path.join(
                folder,
                images[0]
            )

            image = cv2.imread(image_path)

            if image is not None:

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                axes[row, col].imshow(image)

        axes[row, col].set_title(
            split + " - " + class_name
        )

        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
