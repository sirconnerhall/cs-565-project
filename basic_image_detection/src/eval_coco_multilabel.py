import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from coco_tfds_pipeline import make_coco_dataset


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


def to_multilabel_batch(num_classes):
    """
    Convert batched COCO labels -> multi-hot vectors.

    Inputs:
        images: [batch, H, W, 3]
        targets["labels"]: [batch, max_boxes]

    Returns:
        images: [batch, H, W, 3]
        multi_hot: [batch, num_classes]
    """
    def _fn(images, targets):
        labels = targets["labels"]  # [batch, max_boxes]
        labels = tf.cast(labels, tf.int32)

        # NOTE:
        # If you change your padding scheme later (e.g., padding with -1),
        # you can mask out padded labels here before one_hot.
        # For now, this assumes labels are valid class indices (including 0).

        one_hot = tf.one_hot(labels, num_classes)  # [batch, max_boxes, num_classes]
        multi_hot = tf.reduce_max(one_hot, axis=1)  # [batch, num_classes]

        return images, multi_hot

    return _fn


def main():
    # ----------------------------
    # Resolve paths & load config
    # ----------------------------
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "coco_multilabel_config.json"
    config = load_config(str(config_path))

    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    val_split = config.get("val_split", "validation[:5%]")
    model_name = config["model_name"]
    models_dir = project_root / config.get("models_dir", "models")

    best_model_path = models_dir / f"{model_name}_best.keras"
    last_model_path = models_dir / f"{model_name}_last.keras"

    print("Project root:", project_root)
    print("Using val split:", val_split)
    print("Looking for model:", best_model_path)

    # ----------------------------
    # Load model
    # ----------------------------
    if best_model_path.exists():
        model_path = best_model_path
    elif last_model_path.exists():
        print(f"Best model not found, falling back to last model: {last_model_path}")
        model_path = last_model_path
    else:
        raise FileNotFoundError(
            f"Could not find model files:\n  {best_model_path}\n  {last_model_path}"
        )

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    model.summary()

    # ----------------------------
    # Build validation dataset
    # ----------------------------
    # You already set TFDS_DATA_DIR in your env / earlier code.
    val_ds_raw, val_info = make_coco_dataset(
        split=val_split,
        batch_size=batch_size,
        image_size=image_size,
    )

    num_classes = val_info.features["objects"]["label"].num_classes
    class_names = val_info.features["objects"]["label"].names
    print("COCO num classes:", num_classes)

    AUTOTUNE = tf.data.AUTOTUNE

    val_ds = (
        val_ds_raw
        .map(to_multilabel_batch(num_classes), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # ----------------------------
    # Evaluate on the full val set
    # ----------------------------
    print("\nEvaluating on validation set...")
    results = model.evaluate(val_ds, return_dict=True)
    print("\nValidation metrics:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # ----------------------------
    # Inspect a few example predictions
    # ----------------------------
    print("\nShowing a few sample predictions vs ground truth (image-level labels)...")
    threshold = 0.5  # probability threshold for considering a class "present"
    num_batches_to_show = 3

    batch_idx = 0
    for images, multi_hot_gt in val_ds.take(num_batches_to_show):
        preds = model(images, training=False).numpy()  # [B, num_classes]
        multi_hot_gt = multi_hot_gt.numpy()  # [B, num_classes]

        batch_size_actual = images.shape[0]
        for i in range(batch_size_actual):
            gt_vec = multi_hot_gt[i]
            pred_vec = preds[i]

            gt_indices = np.where(gt_vec > 0.5)[0]
            pred_indices = np.where(pred_vec >= threshold)[0]

            gt_labels = [class_names[j] for j in gt_indices]
            pred_labels = [class_names[j] for j in pred_indices]

            print(f"\nBatch {batch_idx}, example {i}:")
            print("  GT labels:   ", gt_labels if gt_labels else "[]")
            print("  Pred labels: ", pred_labels if pred_labels else "[]")

        batch_idx += 1

    print("\nDone. You can tweak 'threshold' or 'num_batches_to_show' in eval_coco_multilabel.py for more detail.")


if __name__ == "__main__":
    main()
