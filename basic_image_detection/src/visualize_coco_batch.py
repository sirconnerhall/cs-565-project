import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

from coco_tfds_pipeline import make_coco_dataset


def draw_image_with_boxes(image, bboxes, ax=None):
    """
    image: [H, W, 3], pixel values [0, 1]
    bboxes: [N, 4] in [ymin, xmin, ymax, xmax] normalized coords
    """
    if ax is None:
        _, ax = plt.subplots(1)

    h, w, _ = image.shape

    ax.imshow(image)
    for box in bboxes:
        ymin, xmin, ymax, xmax = box
        x = xmin * w
        y = ymin * h
        width = (xmax - xmin) * w
        height = (ymax - ymin) * h

        rect = patches.Rectangle(
            (x, y),
            width,
            height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.axis("off")


def main():
    # os.environ["TFDS_DATA_DIR"] = "D:/tensorflow_datasets"  # if needed

    image_size = (224, 224)
    batch_size = 4

    ds, info = make_coco_dataset(
        split="train[:1%]",
        batch_size=batch_size,
        image_size=image_size,
    )

    class_names = info.features["objects"]["label"].names
    print("COCO classes:", len(class_names))

    for images, targets in ds.take(1):
        # images: [B, H, W, 3] normalized 0-1 (because your preprocess divides by 255)
        bboxes = targets["bboxes"]  # [B, max_boxes, 4]
        labels = targets["labels"]  # [B, max_boxes]

        b = 0  # visualize first element in batch
        img = images[b].numpy()
        boxes = bboxes[b].numpy()
        lbls = labels[b].numpy()

        # Filter out zero-size boxes if any
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        lbls = lbls[valid]

        fig, ax = plt.subplots(1, figsize=(6, 6))
        draw_image_with_boxes(img, boxes, ax=ax)

        # Optional: print labels for that image
        unique_labels = sorted(set(lbls.tolist()))
        print("Image labels:", [class_names[i] for i in unique_labels])

        plt.show()
        break


if __name__ == "__main__":
    main()
