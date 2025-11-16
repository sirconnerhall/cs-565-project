"""
Evaluate simple grid-based COCO detector.
- Computes objectness accuracy over the validation set
- Decodes predictions into bounding boxes
- Visualizes predictions vs ground truth
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from coco_tfds_pipeline import make_coco_dataset
from cct_pipeline import make_cct_dataset

from train_simple_detector import (
    infer_grid_size,
    detection_loss,
    objectness_accuracy,
)


# ----------------------------------------------------------
# Helper: decode grid → list of predicted boxes + classes
# ----------------------------------------------------------

def decode_predictions(grid_pred, num_classes, threshold=0.5):
    """
    Input  grid_pred: [S, S, 5 + C]
    Output: list of dicts:
        {
          "bbox": [ymin, xmin, ymax, xmax],
          "class_id": int,
          "score": float
        }
    """

    S = grid_pred.shape[0]
    boxes = []

    for gy in range(S):
        for gx in range(S):
            cell = grid_pred[gy, gx]

            obj = cell[0]
            if obj < threshold:
                continue

            cx, cy, w, h = cell[1:5]

            # Convert from center format
            xmin = cx - w / 2.0
            ymin = cy - h / 2.0
            xmax = cx + w / 2.0
            ymax = cy + h / 2.0

            # Clip to [0, 1]
            xmin = np.clip(xmin, 0, 1)
            ymin = np.clip(ymin, 0, 1)
            xmax = np.clip(xmax, 0, 1)
            ymax = np.clip(ymax, 0, 1)

            # class
            class_probs = cell[5:]
            class_id = int(np.argmax(class_probs))
            score = float(obj)

            boxes.append({
                "bbox": [ymin, xmin, ymax, xmax],
                "class_id": class_id,
                "score": score,
            })

    return boxes


# ----------------------------------------------------------
# Draw helper
# ----------------------------------------------------------

def draw_boxes(image, gt_boxes, pred_boxes, class_names):
    """
    image: [H, W, 3]
    gt_boxes: list of [ymin, xmin, ymax, xmax]
    pred_boxes: same
    """

    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.imshow(image)
    h, w, _ = image.shape

    # Ground truth (green)
    for (ymin, xmin, ymax, xmax) in gt_boxes:
        rect = patches.Rectangle(
            (xmin*w, ymin*h),
            (xmax-xmin)*w,
            (ymax-ymin)*h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)

    # Predictions (red)
    for p in pred_boxes:
        (ymin, xmin, ymax, xmax) = p["bbox"]
        rect = patches.Rectangle(
            (xmin*w, ymin*h),
            (xmax-xmin)*w,
            (ymax-ymin)*h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            xmin*w,
            ymin*h,
            class_names[p["class_id"]],
            color="red",
            fontsize=8,
            bbox=dict(facecolor="yellow", alpha=0.5),
        )

    ax.axis("off")
    plt.show()


# ----------------------------------------------------------
# Load config
# ----------------------------------------------------------

def load_config():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "coco_multilabel_config.json"
    with open(config_path, "r") as f:
        return json.load(f), project_root


# ----------------------------------------------------------
# Main eval loop
# ----------------------------------------------------------

def main():
    config, project_root = load_config()

    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    val_split = config["val_split"]

    model_name = config.get("detector_model_name", "coco_simple_detector")
    models_dir = project_root / config["models_dir"]
    model_path = models_dir / f"{model_name}_best.keras"

    print("Loading model:", model_path)
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "detection_loss": detection_loss,
            "objectness_accuracy": objectness_accuracy,
        }
    )

    model.summary()

    # Build dataset
    dataset_name = config.get("dataset", "coco")

    if dataset_name == "coco":
        val_ds_raw, val_info = make_coco_dataset(
            split=val_split,
            batch_size=batch_size,
            image_size=image_size,
        )
    elif dataset_name == "cct":
        # For CCT, you can use separate filelists for train/val, or the same filelist
        # If using the same filelist, you'll need to split it manually or use a different approach
        val_filelist = config.get("cct_val_file") or config.get("cct_sample_file")
        

        val_ds_raw, val_info = make_cct_dataset(
            images_root=config["cct_images_root"],
            metadata_path=config["cct_annotations"],
            bboxes_path=config["cct_bb_annotations"],
            filelist_path=val_filelist,
            split="val",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_classes = val_info.features["objects"]["label"].num_classes
    class_names = val_info.features["objects"]["label"].names
    grid_size = infer_grid_size(image_size)

    # We need the encoder from train script
    from train_simple_detector import make_grid_encoder
    encoder = make_grid_encoder(num_classes, grid_size)

    AUTOTUNE = tf.data.AUTOTUNE
    val_ds = val_ds_raw.map(encoder).prefetch(AUTOTUNE)

    # ----------------------------------------------------------
    # 1) Numeric evaluation (objectness accuracy)
    # ----------------------------------------------------------

    print("\n=== Running numeric evaluation on validation set ===")
    result = model.evaluate(val_ds, return_dict=True)
    print(result)

    # ----------------------------------------------------------
    # 2) Visualization
    # ----------------------------------------------------------

    print("\n=== Visualizing a few predictions ===")

    for (images, grid_true) in val_ds.take(3):
        preds = model(images, training=False).numpy()

        B = images.shape[0]

        for i in range(B):
            image = images[i].numpy()
            grid_t = grid_true[i].numpy()
            grid_p = preds[i]

            # Extract ground-truth boxes
            gt_boxes = []
            S = grid_size
            for gy in range(S):
                for gx in range(S):
                    cell = grid_t[gy, gx]
                    if cell[0] < 0.5:
                        continue
                    cx, cy, w, h = cell[1:5]
                    xmin = cx - w/2
                    ymin = cy - h/2
                    xmax = cx + w/2
                    ymax = cy + h/2
                    gt_boxes.append([ymin, xmin, ymax, xmax])

            pred_boxes = decode_predictions(grid_p, num_classes, threshold=0.5)

            draw_boxes(image, gt_boxes, pred_boxes, class_names)

        break  # visualizing one batch is enough for now


if __name__ == "__main__":
    main()
