import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers  # works with tf.keras in TF 2.10

from coco_tfds_pipeline import make_coco_dataset  # existing loader
from cct_pipeline import make_cct_dataset
from coco_multilabel_utils import run_extended_sanity_checks

# ----------------------------
# Config helpers
# ----------------------------

def load_config(config_name="coco_multilabel_config.json"):
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / config_name
    with open(config_path, "r") as f:
        config = json.load(f)
    return config, project_root


# ----------------------------
# Infer grid size from backbone
# ----------------------------

def infer_grid_size(image_size):
    """
    Use a MobileNetV2 backbone (no weights) to infer the spatial resolution
    of the final feature map, given the input image size.

    For typical sizes (e.g. 224x224), this will be image_size / 32.
    """
    h, w = image_size
    dummy_input = keras.Input(shape=(h, w, 3))
    base_model = keras.applications.MobileNetV2(
        input_shape=(h, w, 3),
        include_top=False,
        weights=None,  # we just need the shape here
    )

    x = keras.applications.mobilenet_v2.preprocess_input(dummy_input)
    feat = base_model(x)
    grid_h = int(feat.shape[1])
    grid_w = int(feat.shape[2])
    assert grid_h == grid_w, "Expected square feature map for square inputs."
    return grid_h


# ----------------------------
# Encode COCO boxes -> grid targets
# ----------------------------

def make_grid_encoder(num_classes, grid_size):
    """
    Map (images, targets) -> (images, grid_targets), where:

        grid_targets: [B, S, S, 5 + C]
            0: objectness (0 or 1)
            1: cx (normalized 0-1)
            2: cy
            3: w
            4: h
            5..: one-hot class (length C)

    We use tf.numpy_function for clarity: fine for a course project.
    """

    depth = 5 + num_classes

    def _tf_fn(images, targets):
        # images: [B, H, W, 3]
        # targets["bboxes"]: [B, max_boxes, 4]
        # targets["labels"]: [B, max_boxes]

        def _np_encode(images_np, bboxes_np, labels_np):
            """
            NumPy-side implementation over the batch.
            """
            B = images_np.shape[0]
            grid = np.zeros((B, grid_size, grid_size, depth), dtype=np.float32)

            for b in range(B):
                boxes = bboxes_np[b]      # [max_boxes, 4]
                classes = labels_np[b]    # [max_boxes]

                for box, cls in zip(boxes, classes):
                    ymin, xmin, ymax, xmax = box

                    # Skip padded / invalid boxes (zero area)
                    if ymax <= ymin or xmax <= xmin:
                        continue

                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    w = xmax - xmin
                    h = ymax - ymin

                    if w <= 0.0 or h <= 0.0:
                        continue

                    gx = int(cx * grid_size)
                    gy = int(cy * grid_size)

                    if gx < 0 or gx >= grid_size or gy < 0 or gy >= grid_size:
                        continue

                    cls_int = int(cls)
                    if cls_int < 0 or cls_int >= num_classes:
                        continue

                    # If there's already an object in this cell, keep the larger one
                    existing_obj = grid[b, gy, gx, 0]
                    if existing_obj == 1.0:
                        existing_w = grid[b, gy, gx, 3]
                        existing_h = grid[b, gy, gx, 4]
                        if w * h <= existing_w * existing_h:
                            continue

                    # Set objectness
                    grid[b, gy, gx, 0] = 1.0
                    # Bbox (normalized)
                    grid[b, gy, gx, 1:5] = [cx, cy, w, h]
                    # One-hot class
                    grid[b, gy, gx, 5 + cls_int] = 1.0

            return grid

        # Ensure labels are int32 for consistency (CCT uses int32, COCO uses int64)
        labels = tf.cast(targets["labels"], tf.int32)
        
        grid = tf.numpy_function(
            _np_encode,
            [images, targets["bboxes"], labels],
            tf.float32,
        )

        # Set dynamic batch dim, fixed spatial dims and depth
        grid.set_shape((None, grid_size, grid_size, depth))
        return images, grid

    return _tf_fn


# ----------------------------
# Simple detector model
# ----------------------------

def build_simple_detector(image_size, num_classes):
    """
    Simple grid-based detector on top of MobileNetV2.
    Output shape: [B, S, S, 5 + C] with sigmoid activation.
    """

    h, w = image_size
    inputs = keras.Input(shape=(h, w, 3), name="image")

    base_model = keras.applications.MobileNetV2(
        input_shape=(h, w, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # start frozen; could fine-tune later

    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)

    # A small conv head
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    # 5 + num_classes channels: [obj, cx, cy, w, h, class_probs...]
    outputs = layers.Conv2D(
        5 + num_classes,
        1,
        padding="same",
        activation="sigmoid",  # all outputs in [0, 1]
        name="grid_output",
    )(x)

    model = keras.Model(inputs, outputs, name="coco_simple_detector")
    return model


# ----------------------------
# Detection loss
# ----------------------------

def detection_loss(y_true, y_pred):
    """
    Combined loss:
      - objectness BCE for all cells
      - bbox MSE for cells with an object
      - class BCE for cells with an object

    Shapes:
        y_true, y_pred: [B, S, S, 5 + C]
          0: objectness
          1: cx
          2: cy
          3: w
          4: h
          5..: class one-hot (C)
    """

    # Objectness
    obj_true = y_true[..., 0:1]   # [B, S, S, 1]
    obj_pred = y_pred[..., 0:1]   # [B, S, S, 1]

    # BCE returns [B, S, S, 1]
    obj_loss = tf.keras.backend.binary_crossentropy(obj_true, obj_pred)

    # Bboxes (only where there is an object)
    box_true = y_true[..., 1:5]   # [B, S, S, 4]
    box_pred = y_pred[..., 1:5]   # [B, S, S, 4]

    # Manual MSE per coordinate
    box_diff = box_true - box_pred               # [B, S, S, 4]
    box_sq = tf.square(box_diff)                 # [B, S, S, 4]
    box_loss = tf.reduce_sum(box_sq, axis=-1, keepdims=True)  # [B, S, S, 1]

    # Mask by objectness (only cells with objects matter)
    box_loss = box_loss * obj_true               # [B, S, S, 1]

    # Classes (only where there is an object)
    cls_true = y_true[..., 5:]   # [B, S, S, C]
    cls_pred = y_pred[..., 5:]   # [B, S, S, C]

    # BCE per class → [B, S, S, C]
    cls_bce = tf.keras.backend.binary_crossentropy(cls_true, cls_pred)

    # Average over classes → [B, S, S, 1]
    cls_loss = tf.reduce_mean(cls_bce, axis=-1, keepdims=True)

    # Mask by objectness
    cls_loss = cls_loss * obj_true               # [B, S, S, 1]

    # Sum all three terms; shapes all [B, S, S, 1]
    total = obj_loss + box_loss + cls_loss

    return tf.reduce_mean(total)

# ----------------------------
# Simple objectness accuracy metric
# ----------------------------

def objectness_accuracy(y_true, y_pred):
    obj_true = y_true[..., 0]
    obj_pred = y_pred[..., 0]
    obj_pred_label = tf.cast(obj_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(obj_true, obj_pred_label), tf.float32))


# ----------------------------
# Main training loop
# ----------------------------

def main():
    # Load config (reuse your existing coco_multilabel_config.json)
    config, project_root = load_config("coco_multilabel_config.json")

    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    train_split = config["train_split"]
    val_split = config["val_split"]
    learning_rate = config["learning_rate"]
    models_dir = project_root / config["models_dir"]

    # Optional: detector-specific model name in config; else fallback
    model_name = config.get("detector_model_name", "coco_simple_detector")

    models_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", project_root)
    print("Train split:", train_split)
    print("Val split:  ", val_split)
    print("Image size: ", image_size)

    # Use your existing COCO pipeline (with padded boxes & labels)
    
    dataset_name = config.get("dataset", "coco")

    if dataset_name == "coco":
        train_ds_raw, train_info = make_coco_dataset(
            split=train_split,
            batch_size=batch_size,
            image_size=image_size,
        )
        val_ds_raw, val_info = make_coco_dataset(
            split=val_split,
            batch_size=batch_size,
            image_size=image_size,
        )
    elif dataset_name == "cct":
        # For CCT, you can use separate filelists for train/val, or the same filelist
        # If using the same filelist, you'll need to split it manually or use a different approach
        train_filelist = config.get("cct_train_file") or config.get("cct_sample_file")
        val_filelist = config.get("cct_val_file") or config.get("cct_sample_file")
        
        train_ds_raw, train_info = make_cct_dataset(
            images_root=config["cct_images_root"],
            metadata_path=config["cct_annotations"],
            bboxes_path=config["cct_bb_annotations"],
            filelist_path=train_filelist,
            split="train",
            batch_size=batch_size,
            image_size=image_size,
        )

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
        
        # Ensure both datasets use the same num_classes (from train_info)
        # This is important because both should have the same category mapping
        if train_info.features["objects"]["label"].num_classes != val_info.features["objects"]["label"].num_classes:
            raise ValueError(
                f"Train and val datasets have different num_classes: "
                f"{train_info.features['objects']['label'].num_classes} vs "
                f"{val_info.features['objects']['label'].num_classes}"
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_classes = train_info.features["objects"]["label"].num_classes
    print(f"{dataset_name.upper()} num classes (detector):", num_classes)

    # Infer grid size from the backbone stride
    grid_size = infer_grid_size(image_size)
    print("Grid size (SxS):", grid_size, "x", grid_size)

    AUTOTUNE = tf.data.AUTOTUNE

    encoder = make_grid_encoder(num_classes, grid_size)

    train_ds = (
        train_ds_raw
        .map(encoder, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds_raw
        .map(encoder, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # Build model
    model = build_simple_detector(image_size, num_classes)
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=detection_loss,
        metrics=[objectness_accuracy],
    )

    best_model_path = models_dir / f"{model_name}_best.keras"
    last_model_path = models_dir / f"{model_name}_last.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            mode="min",
            verbose=1,
        ),
    ]
    
    # Extended sanity check
    run_extended_sanity_checks(
        train_ds_raw=train_ds_raw,
        train_ds=train_ds,
        num_classes=num_classes,
        image_size=image_size,
        grid_size=grid_size,
        model=model,
    )

    print("\n=== Training simple grid-based detector ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save(str(last_model_path))
    print(f"\nSaved last model to {last_model_path}")
    print(f"Best model (by val_loss) at {best_model_path}")


if __name__ == "__main__":
    main()
