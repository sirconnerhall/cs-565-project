"""
Train YOLO-style detection model with metadata integration for CCT dataset.

Uses pre-trained backbone and integrates CCT metadata (location, date, time).
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ..pipelines.cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from .build_detector import build_ssd_detector_with_metadata
from ..utils.detection_utils import (
    DetectionLossFocal,
    make_component_loss_metrics,
    objectness_accuracy,
    PredictionStats,
)


def load_config(config_name="config.json"):
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / config_name
    with open(config_path, "r") as f:
        config = json.load(f)
    return config, project_root


def make_grid_encoder_for_detector(num_classes, image_size, grid_size=None):
    """
    Create encoder that converts CCT format to grid format for detector.
    Uses tf.numpy_function to allow Python loops (same pattern as existing encoder).
    
    Args:
        num_classes: Number of classes
        image_size: (height, width) of input images
        grid_size: (height, width) of output grid. If None, estimates from image_size.
    """
    h, w = image_size
    
    # Use provided grid_size or estimate
    if grid_size is not None:
        if isinstance(grid_size, (tuple, list)):
            grid_h, grid_w = grid_size
        else:
            grid_h = grid_w = grid_size
    else:
        # Estimate grid size based on input size and backbone downsampling
        # MobileNetV2 downsamples by ~32x, so 224/32 ≈ 7
        if h == 224 and w == 224:
            grid_h = grid_w = 7  # Typical for MobileNetV2
        elif h == 320 and w == 320:
            grid_h = grid_w = 10  # 320/32 = 10
        else:
            # Rough estimate: divide by 32 (typical downsampling)
            grid_h = grid_w = max(7, int(h / 32))
    
    depth = 1 + 4 + num_classes  # objectness + bbox + classes
    
    def _tf_fn(images, targets):
        """
        Encode targets to grid format using numpy function.
        
        Args:
            images: [B, H, W, 3]
            targets: Dict with "bboxes" [B, N, 4] and "labels" [B, N]
        
        Returns:
            images, grid_targets: [B, grid_size, grid_size, 1 + 4 + num_classes]
        """
        def _np_encode(images_np, bboxes_np, labels_np):
            """
            NumPy-side implementation over the batch.
            """
            B = images_np.shape[0]
            grid = np.zeros((B, grid_h, grid_w, depth), dtype=np.float32)
            
            for b in range(B):
                boxes = bboxes_np[b]      # [max_boxes, 4] in [ymin, xmin, ymax, xmax]
                classes = labels_np[b]    # [max_boxes]
                
                for box, cls in zip(boxes, classes):
                    ymin, xmin, ymax, xmax = box
                    
                    # Skip padded / invalid boxes (zero area)
                    if ymax <= ymin or xmax <= xmin:
                        continue
                    
                    # Convert to center format (normalized)
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    w = xmax - xmin
                    h = ymax - ymin
                    
                    if w <= 0.0 or h <= 0.0:
                        continue
                    
                    # Find grid cell
                    gx = int(cx * grid_w)
                    gy = int(cy * grid_h)
                    
                    if gx < 0 or gx >= grid_w or gy < 0 or gy >= grid_h:
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
                    # Bbox (normalized cx, cy, w, h)
                    grid[b, gy, gx, 1:5] = [cx, cy, w, h]
                    # One-hot class
                    grid[b, gy, gx, 5 + cls_int] = 1.0
            
            return grid
        
        # Ensure labels are int32 for consistency
        labels = tf.cast(targets["labels"], tf.int32)
        
        grid = tf.numpy_function(
            _np_encode,
            [images, targets["bboxes"], labels],
            tf.float32,
        )
        
        # Set dynamic batch dim, fixed spatial dims and depth
        grid.set_shape((None, grid_h, grid_w, depth))
        return images, grid
    
    return _tf_fn


def main():
    config, project_root = load_config("config.json")
    
    # Check if using pretrained detector
    use_pretrained = config.get("use_pretrained_detector", False)
    if not use_pretrained:
        print("[Warning] use_pretrained_detector is False. Use train_cct_multimodal_detector.py instead.")
        return
    
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    models_dir = project_root / config["models_dir"]
    # Model name is set here, not from config
    model_name = "TSM"
    
    pretrained_model_type = config.get("pretrained_model_type", "ssd_mobilenet_v2")
    freeze_backbone_epochs = config.get("freeze_backbone_epochs", 5)
    
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 4.0)
    bbox_loss_weight = config.get("bbox_loss_weight", 10.0)
    filter_empty_images = config.get("filter_empty_images", False)
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Training TSM Detector (Two Stage, with Metadata)")
    print("=" * 60)
    print(f"Model type: {pretrained_model_type}")
    print(f"Image size: {image_size}")
    print(f"Freeze backbone for: {freeze_backbone_epochs} epochs")
    
    # Load CCT dataset
    dataset_name = config.get("dataset", "cct")
    metadata_dim = 5
    
    if dataset_name != "cct":
        raise ValueError(f"TSM detector currently only supports CCT dataset, got: {dataset_name}")
    
    # Load datasets from TFRecords
    cct_tfrecords_dir = config.get("cct_tfrecords_dir")
    if not cct_tfrecords_dir or not Path(cct_tfrecords_dir).exists():
        raise ValueError(f"TFRecords directory not found: {cct_tfrecords_dir}. Please generate TFRecords first.")
    
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
    
    # Add placeholder metadata for TFRecords
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
    
    # Build model first to get actual output shape
    print("\n[Model] Building detector...")
    model = build_ssd_detector_with_metadata(
        image_size=image_size,
        num_classes=num_classes,
        metadata_dim=metadata_dim,
        backbone_type=pretrained_model_type,
        freeze_backbone=True,  # Start frozen
    )
    
    # Get actual output shape from model
    # Do a test forward pass to determine output dimensions
    test_image = tf.zeros((1, image_size[0], image_size[1], 3))
    test_metadata = tf.zeros((1, metadata_dim))
    test_output = model([test_image, test_metadata], training=False)
    output_shape = test_output.shape
    grid_h, grid_w = output_shape[1], output_shape[2]
    print(f"[Model] Detected output grid size: {grid_h}x{grid_w}")
    
    model.summary()
    
    # Create encoder with the correct grid size
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
    
    # Callbacks
    best_model_path = models_dir / f"{model_name}_best.keras"
    last_model_path = models_dir / f"{model_name}_last.keras"
    
    # Create callbacks for Stage 1
    callbacks_stage1 = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,  # More patience to allow Stage 2 to run
            mode="min",
            restore_best_weights=False,  # Don't restore, let Stage 2 continue
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            mode="min",
            verbose=1,
        ),
        PredictionStats(threshold=0.5, val_dataset=val_ds, num_classes=num_classes, grid_size=grid_h),
    ]
    
    # Callbacks for Stage 2 (same but with restored best weights)
    callbacks_stage2 = [
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
        PredictionStats(threshold=0.5, val_dataset=val_ds, num_classes=num_classes, grid_size=grid_h),
    ]
    
    # Stage 1: Train with frozen backbone
    print("\n" + "=" * 60)
    print(f"Stage 1: Training with frozen backbone ({freeze_backbone_epochs} epochs)")
    print(f"  Total epochs configured: {epochs}")
    print("=" * 60)
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=freeze_backbone_epochs,
        callbacks=callbacks_stage1,
        verbose=1,
    )
    
    # Stage 2: Unfreeze and fine-tune
    remaining_epochs = epochs - freeze_backbone_epochs
    print(f"\n[Stage 2 Check] Total epochs: {epochs}, Stage 1 completed: {freeze_backbone_epochs}, Remaining: {remaining_epochs}")
    
    if remaining_epochs > 0:
        print("\n" + "=" * 60)
        print(f"Stage 2: Fine-tuning with unfrozen backbone ({remaining_epochs} epochs)")
        print("=" * 60)
        
        # Unfreeze backbone - need to find the actual backbone model
        trainable_count = 0
        for layer in model.layers:
            layer_name_lower = layer.name.lower()
            if ("backbone" in layer.name or 
                "mobilenet" in layer_name_lower or 
                "resnet" in layer_name_lower or
                "efficientnet" in layer_name_lower):
                # Check if it's a model (sub-model) or a layer
                if isinstance(layer, keras.Model):
                    # Unfreeze all layers in the sub-model
                    for sub_layer in layer.layers:
                        sub_layer.trainable = True
                        trainable_count += 1
                else:
                    layer.trainable = True
                    trainable_count += 1
        
        print(f"[Stage 2] Unfroze {trainable_count} backbone layers")
        
        # Lower learning rate for fine-tuning (but slightly higher for small datasets)
        fine_tune_lr = config.get("fine_tune_lr", learning_rate * 0.1)
        print(f"[Stage 2] Using learning rate: {fine_tune_lr} (Stage 1 used: {learning_rate})")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss=loss_fn,
            metrics=[objectness_accuracy] + component_metrics,
        )
        
        print(f"[Stage 2] Starting training from epoch {freeze_backbone_epochs} to {epochs}")
        history_stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            initial_epoch=freeze_backbone_epochs,
            callbacks=callbacks_stage2,
            verbose=1,
        )
        print(f"[Stage 2] Completed training from epoch {freeze_backbone_epochs} to {epochs}")
    else:
        print(f"\n[Info] No Stage 2 training needed (epochs={epochs} <= freeze_backbone_epochs={freeze_backbone_epochs})")
    
    model.save(str(last_model_path))
    print(f"\nSaved model to {last_model_path}")
    print(f"Best model (by val_loss) at {best_model_path}")


if __name__ == "__main__":
    main()

