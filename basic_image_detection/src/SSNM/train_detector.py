"""
Train single-stage detector without metadata (SSNM).
Backbone stays frozen throughout training.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from ..pipelines.coco_tfds_pipeline import make_coco_dataset
from ..pipelines.cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from ..pipelines.coco_multilabel_utils import run_extended_sanity_checks
from ..utils.detection_utils import (
    DetectionLossFocal,
    make_component_loss_metrics,
    objectness_accuracy,
)
from .build_detector import build_detector


def load_config(config_name="config.json"):
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / config_name
    with open(config_path, "r") as f:
        config = json.load(f)
    return config, project_root


def infer_grid_size(image_size):
    """Infer grid size from backbone stride."""
    h, w = image_size
    dummy_input = keras.Input(shape=(h, w, 3))
    base_model = keras.applications.MobileNetV2(
        input_shape=(h, w, 3),
        include_top=False,
        weights=None,
    )
    x = keras.applications.mobilenet_v2.preprocess_input(dummy_input)
    feat = base_model(x)
    grid_h = int(feat.shape[1])
    grid_w = int(feat.shape[2])
    assert grid_h == grid_w, "Expected square feature map for square inputs."
    return grid_h


def make_grid_encoder(num_classes, grid_size):
    """Map (images, targets) -> (images, grid_targets)."""
    depth = 5 + num_classes

    def _tf_fn(images, targets):
        def _np_encode(images_np, bboxes_np, labels_np):
            B = images_np.shape[0]
            grid = np.zeros((B, grid_size, grid_size, depth), dtype=np.float32)

            for b in range(B):
                boxes = bboxes_np[b]
                classes = labels_np[b]

                for box, cls in zip(boxes, classes):
                    ymin, xmin, ymax, xmax = box

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

                    existing_obj = grid[b, gy, gx, 0]
                    if existing_obj == 1.0:
                        existing_w = grid[b, gy, gx, 3]
                        existing_h = grid[b, gy, gx, 4]
                        if w * h <= existing_w * existing_h:
                            continue

                    grid[b, gy, gx, 0] = 1.0
                    grid[b, gy, gx, 1:5] = [cx, cy, w, h]
                    grid[b, gy, gx, 5 + cls_int] = 1.0

            return grid

        labels = tf.cast(targets["labels"], tf.int32)
        grid = tf.numpy_function(
            _np_encode,
            [images, targets["bboxes"], labels],
            tf.float32,
        )
        grid.set_shape((None, grid_size, grid_size, depth))
        return images, grid

    return _tf_fn


class PredictionStats(keras.callbacks.Callback):
    """Callback to log detailed prediction statistics during training."""
    def __init__(self, threshold=0.5, val_dataset=None, num_classes=None, grid_size=None):
        super().__init__()
        self.threshold = threshold
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.grid_size = grid_size
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"\n{'='*60}")
        print(f"[Epoch {epoch+1}] Training Metrics:")
        print(f"{'='*60}")
        print(f"  Loss: {logs.get('loss', 'N/A'):.4f}, Val Loss: {logs.get('val_loss', 'N/A'):.4f}")
        print(f"  Objectness Acc: {logs.get('objectness_accuracy', 'N/A'):.4f}, Val: {logs.get('val_objectness_accuracy', 'N/A'):.4f}")
        
        if 'objectness_loss' in logs:
            print(f"  Loss Components:")
            print(f"    Objectness: {logs.get('objectness_loss', 'N/A'):.4f} (val: {logs.get('val_objectness_loss', 'N/A'):.4f})")
            print(f"    BBox: {logs.get('bbox_loss', 'N/A'):.4f} (val: {logs.get('val_bbox_loss', 'N/A'):.4f})")
            print(f"    Class: {logs.get('class_loss', 'N/A'):.4f} (val: {logs.get('val_class_loss', 'N/A'):.4f})")
        
        print(f"{'='*60}")


def main():
    config, project_root = load_config("config.json")

    # Model name is set here, not from config
    model_name = "SSNM"
    
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    train_split = config["train_split"]
    val_split = config["val_split"]
    learning_rate = config["learning_rate"]
    models_dir = project_root / config["models_dir"]

    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 5.0)
    bbox_loss_weight = config.get("bbox_loss_weight", 10.0)
    filter_empty_images = config.get("filter_empty_images", False)
    use_focal_loss = config.get("use_focal_loss", True)

    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training SSNM Detector (Single Stage, No Metadata)")
    print("=" * 60)
    print(f"Model name: {model_name}")
    print(f"Image size: {image_size}")
    print(f"Backbone: Frozen (single-stage transfer learning)")

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
        cct_tfrecords_dir = config.get("cct_tfrecords_dir")
        if not cct_tfrecords_dir or not Path(cct_tfrecords_dir).exists():
            raise ValueError(f"TFRecords directory not found: {cct_tfrecords_dir}. Please generate TFRecords first.")
        
        print(f"[CCT] Using TFRecords from {cct_tfrecords_dir}")
        train_ds_raw, train_info = make_cct_tfrecords_dataset(
            tfrecords_dir=cct_tfrecords_dir,
            split="train",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
        )
        val_ds_raw, val_info = make_cct_tfrecords_dataset(
            tfrecords_dir=cct_tfrecords_dir,
            split="val",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_classes = train_info.features["objects"]["label"].num_classes
    print(f"{dataset_name.upper()} num classes: {num_classes}")

    grid_size = infer_grid_size(image_size)
    print(f"Grid size: {grid_size}x{grid_size}")

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
    model = build_detector(image_size, num_classes)
    model.summary()

    # Compile
    if use_focal_loss:
        loss_fn = DetectionLossFocal(
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            positive_weight=positive_weight,
            bbox_loss_weight=bbox_loss_weight
        )
        component_metrics = make_component_loss_metrics(loss_fn)
        metrics = [objectness_accuracy] + component_metrics
    else:
        from ..utils.detection_utils import detection_loss
        loss_fn = detection_loss
        metrics = [objectness_accuracy]
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=metrics,
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
            patience=5,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            mode="min",
            verbose=1,
        ),
        PredictionStats(threshold=0.5, val_dataset=val_ds, num_classes=num_classes, grid_size=grid_size),
    ]
    
    if dataset_name == "coco":
        run_extended_sanity_checks(
            train_ds_raw=train_ds_raw,
            train_ds=train_ds,
            num_classes=num_classes,
            image_size=image_size,
            grid_size=grid_size,
            model=model,
        )

    print("\n=== Training SSNM Detector ===")
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

