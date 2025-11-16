import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from coco_tfds_pipeline import make_coco_dataset


# ----------------------------
# Config helpers
# ----------------------------

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


# ----------------------------
# COCO labels -> multi-hot
# ----------------------------
from coco_multilabel_utils import to_multilabel_batch

# ----------------------------
# Synthetic metadata
# ----------------------------

def add_synthetic_metadata(num_locations=4):
    """
    Input:  (images, multi_hot_labels)
    Output: ((images, metadata_vec), multi_hot_labels)

    metadata = [brightness, is_day, is_night, one_hot_location(num_locations)]
            -> dim = 3 + num_locations
    """
    def _fn(images, multi_hot):
        # images: [B, H, W, 3], values in [0,1]
        batch_size = tf.shape(images)[0]

        # Brightness per image
        brightness = tf.reduce_mean(images, axis=[1, 2, 3])  # [B]
        brightness = tf.expand_dims(brightness, -1)          # [B,1]

        # Day/night flags based on brightness
        is_day = tf.cast(brightness[:, 0] > 0.5, tf.float32)  # [B]
        is_night = 1.0 - is_day

        is_day = tf.expand_dims(is_day, -1)       # [B,1]
        is_night = tf.expand_dims(is_night, -1)   # [B,1]

        # Fake location ID (0..num_locations-1)
        loc_ids = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=num_locations,
            dtype=tf.int32,
        )
        loc_onehot = tf.one_hot(loc_ids, num_locations)  # [B, num_locations]

        metadata = tf.concat(
            [brightness, is_day, is_night, loc_onehot],
            axis=1,
        )  # [B, 3 + num_locations]

        return (images, metadata), multi_hot

    return _fn


def main():
    # ----------------------------
    # Paths & config
    # ----------------------------
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "coco_multilabel_config.json"
    config = load_config(str(config_path))

    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    val_split = config.get("val_split", "validation[:5%]")
    model_name = config.get("model_name", "coco_multimodal_mobilenetv2")
    models_dir = project_root / config.get("models_dir", "models")

    num_locations = 4
    metadata_dim = 3 + num_locations  # must match training script

    best_model_path = models_dir / f"{model_name}_best.keras"
    last_model_path = models_dir / f"{model_name}_last.keras"

    print("Project root:", project_root)
    print("Val split:   ", val_split)
    print("Looking for model:", best_model_path)

    # ----------------------------
    # Load model (ignore original loss)
    # ----------------------------
    if best_model_path.exists():
        model_path = best_model_path
    elif last_model_path.exists():
        print(f"[WARN] Best model not found, using last model: {last_model_path}")
        model_path = last_model_path
    else:
        raise FileNotFoundError(
            f"Could not find model files:\n  {best_model_path}\n  {last_model_path}"
        )

    # We don't need the original focal loss to run eval; recompile later.
    print("Loading model (compile=False) from:", model_path)
    model = keras.models.load_model(model_path, compile=False)
    model.summary()

    # ----------------------------
    # Build validation dataset
    # ----------------------------
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
        .map(add_synthetic_metadata(num_locations), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # ----------------------------
    # Compile model for evaluation
    # ----------------------------
    # Use a simple loss here; metrics are what we mainly care about.
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="bin_acc"),
            keras.metrics.AUC(name="auc"),
        ],
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
    threshold = 0.5
    num_batches_to_show = 3

    batch_idx = 0
    for (images, metadata), multi_hot_gt in val_ds.take(num_batches_to_show):
        preds = model([images, metadata], training=False).numpy()  # [B, C]
        multi_hot_gt = multi_hot_gt.numpy()  # [B, C]

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

    print("\nDone.")


if __name__ == "__main__":
    main()
