"""
Train single-stage detector with metadata (SSM).
Backbone stays frozen throughout training.
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..pipelines.cct_tfrecords_pipeline import make_cct_tfrecords_dataset
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


def make_grid_encoder_for_detector(num_classes, image_size, grid_size=None):
    """Create encoder that converts CCT format to grid format."""
    h, w = image_size
    
    if grid_size is not None:
        if isinstance(grid_size, (tuple, list)):
            grid_h, grid_w = grid_size
        else:
            grid_h = grid_w = grid_size
    else:
        if h == 224 and w == 224:
            grid_h = grid_w = 7
        elif h == 320 and w == 320:
            grid_h = grid_w = 10
        else:
            grid_h = grid_w = max(7, int(h / 32))
    
    depth = 1 + 4 + num_classes
    
    def _tf_fn(images, targets):
        def _np_encode(images_np, bboxes_np, labels_np):
            B = images_np.shape[0]
            grid = np.zeros((B, grid_h, grid_w, depth), dtype=np.float32)
            
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
                    
                    gx = int(cx * grid_w)
                    gy = int(cy * grid_h)
                    
                    if gx < 0 or gx >= grid_w or gy < 0 or gy >= grid_h:
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
        grid.set_shape((None, grid_h, grid_w, depth))
        return images, grid
    
    return _tf_fn


def main():
    config, project_root = load_config("config.json")
    
    # Model name is set here, not from config
    model_name = "SSM"
    
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    models_dir = project_root / config["models_dir"]
    
    pretrained_model_type = config.get("pretrained_model_type", "mobilenet_v2")
    
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 4.0)
    bbox_loss_weight = config.get("bbox_loss_weight", 10.0)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Training SSM Detector (Single Stage, with Metadata)")
    print("=" * 60)
    print(f"Model name: {model_name}")
    print(f"Image size: {image_size}")
    print(f"Backbone: Frozen (single-stage transfer learning)")
    
    dataset_name = config.get("dataset", "cct")
    metadata_dim = 5
    
    if dataset_name != "cct":
        raise ValueError(f"SSM detector currently only supports CCT dataset, got: {dataset_name}")
    
    cct_tfrecords_dir = config.get("cct_tfrecords_dir")
    if not cct_tfrecords_dir or not Path(cct_tfrecords_dir).exists():
        raise ValueError(f"TFRecords directory not found: {cct_tfrecords_dir}.")
    
    print(f"[Dataset] Using TFRecords from {cct_tfrecords_dir}")
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
    
    num_classes = train_info.features["objects"]["label"].num_classes
    print(f"Number of classes: {num_classes}")
    
    # Add placeholder metadata
    def add_placeholder_metadata(images, targets):
        batch_size = tf.shape(images)[0]
        brightness = tf.reduce_mean(images, axis=[1, 2, 3])
        location_id = tf.zeros([batch_size], dtype=tf.float32)
        hour = tf.ones([batch_size], dtype=tf.float32) * 0.5
        day_of_week = tf.ones([batch_size], dtype=tf.float32) * 0.5
        month = tf.ones([batch_size], dtype=tf.float32) * 0.5
        metadata = tf.stack([location_id, hour, day_of_week, month, brightness], axis=1)
        return (images, metadata), targets
    
    train_ds_with_meta = train_ds_raw.map(add_placeholder_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_with_meta = val_ds_raw.map(add_placeholder_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Build model
    print("\n[Model] Building detector...")
    model = build_detector(
        image_size=image_size,
        num_classes=num_classes,
        metadata_dim=metadata_dim,
        backbone_type=pretrained_model_type,
    )
    
    # Get output shape
    test_image = tf.zeros((1, image_size[0], image_size[1], 3))
    test_metadata = tf.zeros((1, metadata_dim))
    test_output = model([test_image, test_metadata], training=False)
    output_shape = test_output.shape
    grid_h, grid_w = output_shape[1], output_shape[2]
    print(f"[Model] Detected output grid size: {grid_h}x{grid_w}")
    
    model.summary()
    
    # Create encoder
    encoder = make_grid_encoder_for_detector(num_classes, image_size, grid_size=(grid_h, grid_w))
    
    def encode_with_metadata(inputs, targets):
        images, metadata = inputs
        images_encoded, grid_targets = encoder(images, targets)
        return (images_encoded, metadata), grid_targets
    
    train_ds = train_ds_with_meta.map(encode_with_metadata, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds_with_meta.map(encode_with_metadata, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # Compile
    loss_fn = DetectionLossFocal(
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        positive_weight=positive_weight,
        bbox_loss_weight=bbox_loss_weight
    )
    component_metrics = make_component_loss_metrics(loss_fn)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[objectness_accuracy] + component_metrics,
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
            patience=10,
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
    ]
    
    print("\n=== Training SSM Detector ===")
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

