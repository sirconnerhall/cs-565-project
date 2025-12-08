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
from ..utils.anchor_encoder import make_anchor_encoder_with_metadata


def load_config(config_name="config.json"):
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / config_name
    with open(config_path, "r") as f:
        config = json.load(f)
    return config, project_root


def infer_grid_size(image_size):
    """Infer grid size from CSPDarkNet backbone stride (32x downsampling)."""
    h, w = image_size
    grid_h = h // 32
    grid_w = w // 32
    return grid_h, grid_w


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
    objectness_label_smoothing = config.get("objectness_label_smoothing", 0.1)
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
    metadata_dim = 8  # Updated: 8 features with cyclical encoding
    
    if dataset_name != "cct":
        raise ValueError(f"TSM detector currently only supports CCT dataset, got: {dataset_name}")
    
    # Load datasets from TFRecords
    cct_tfrecords_dir = config.get("cct_tfrecords_dir")
    if not cct_tfrecords_dir or not Path(cct_tfrecords_dir).exists():
        raise ValueError(f"TFRecords directory not found: {cct_tfrecords_dir}. Please generate TFRecords first.")
    
    # Get split names from config (default to "train" and "val" if not specified)
    train_split = config.get("train_split", "train")
    val_split = config.get("val_split", "val")
    
    # Extract base split name (remove percentage syntax like "[:1%]")
    train_split_base = train_split.split("[")[0].strip() if "[" in train_split else train_split
    val_split_base = val_split.split("[")[0].strip() if "[" in val_split else val_split
    
    # Map common split name variations to TFRecord naming convention
    # TFRecord files use "train" and "val", not "training" or "validation"
    split_name_map = {
        "train": "train",
        "training": "train",
        "val": "val",
        "validation": "val",
        "valid": "val",
        "test": "test",
        "testing": "test",
    }
    train_split_base = split_name_map.get(train_split_base.lower(), train_split_base)
    val_split_base = split_name_map.get(val_split_base.lower(), val_split_base)
    
    print(f"[Dataset] Using TFRecords from {cct_tfrecords_dir}")
    print(f"[Dataset] Train split: {train_split_base}, Val split: {val_split_base}")
    train_ds_raw, train_info = make_cct_tfrecords_dataset(
        tfrecords_dir=cct_tfrecords_dir,
        split=train_split_base,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
    )
    val_ds_raw, val_info = make_cct_tfrecords_dataset(
        tfrecords_dir=cct_tfrecords_dir,
        split=val_split_base,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    
    # Apply percentage filtering if specified in config
    if "[" in train_split and "%" in train_split:
        import re
        match = re.search(r'\[:(\d+)%\]', train_split)
        if match:
            percentage = int(match.group(1))
            try:
                total_size = train_info.num_samples if hasattr(train_info, 'num_samples') else None
                if total_size is None:
                    cardinality = train_ds_raw.cardinality().numpy()
                    total_size = cardinality if cardinality >= 0 else None
            except:
                total_size = None
            
            if total_size and total_size > 0:
                take_size = max(1, int(total_size * percentage / 100))
                print(f"[Dataset] Limiting train split to {percentage}% ({take_size} of {total_size} samples)")
                train_ds_raw = train_ds_raw.take(take_size)
            else:
                print(f"[Dataset] Warning: Could not determine dataset size for percentage filtering")
    
    if "[" in val_split and "%" in val_split:
        import re
        match = re.search(r'\[:(\d+)%\]', val_split)
        if match:
            percentage = int(match.group(1))
            try:
                total_size = val_info.num_samples if hasattr(val_info, 'num_samples') else None
                if total_size is None:
                    cardinality = val_ds_raw.cardinality().numpy()
                    total_size = cardinality if cardinality >= 0 else None
            except:
                total_size = None
            
            if total_size and total_size > 0:
                take_size = max(1, int(total_size * percentage / 100))
                print(f"[Dataset] Limiting val split to {percentage}% ({take_size} of {total_size} samples)")
                val_ds_raw = val_ds_raw.take(take_size)
            else:
                print(f"[Dataset] Warning: Could not determine dataset size for percentage filtering")
    
    num_classes = train_info.features["objects"]["label"].num_classes
    print(f"Number of classes: {num_classes}")
    
    # Get grid size and anchor configuration
    grid_h, grid_w = infer_grid_size(image_size)
    grid_size = (grid_h, grid_w)
    num_anchors = config.get("num_anchors", 3)
    print(f"Grid size: {grid_h}x{grid_w}, num_anchors: {num_anchors}")
    
    # Extract and encode actual metadata from TFRecords with cyclical encoding
    from ..utils.metadata_encoding import encode_metadata_from_tfrecords
    
    def add_metadata(images, targets):
        # Extract location and date_captured from targets
        # These should already be batched [B] tensors from the TFRecord parser
        batch_size = tf.shape(images)[0]
        
        # Get location and date_captured, with defaults if missing
        location_str = targets.get("location")
        date_captured_str = targets.get("date_captured")
        
        # Create default values if missing
        if location_str is None:
            location_str = tf.fill([batch_size], "")
        if date_captured_str is None:
            date_captured_str = tf.fill([batch_size], "2000-01-01 12:00:00")
        
        # Ensure they're properly shaped (handle scalar case)
        location_str = tf.reshape(location_str, [batch_size])
        date_captured_str = tf.reshape(date_captured_str, [batch_size])
        
        # Check if location_str is empty and replace with default
        location_str = tf.where(
            tf.equal(tf.strings.length(location_str), 0),
            tf.fill([batch_size], "0"),
            location_str
        )
        
        # Check if date_captured_str is empty and replace with default
        date_captured_str = tf.where(
            tf.equal(tf.strings.length(date_captured_str), 0),
            tf.fill([batch_size], "2000-01-01 12:00:00"),
            date_captured_str
        )
        
        # Encode metadata with cyclical encoding
        metadata = encode_metadata_from_tfrecords(images, location_str, date_captured_str)
        
        # Create new targets dict without metadata strings (keep only bboxes and labels)
        targets_clean = {
            "bboxes": targets["bboxes"],
            "labels": targets["labels"],
        }
        
        return (images, metadata), targets_clean
    
    train_ds_with_meta = train_ds_raw.map(add_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_with_meta = val_ds_raw.map(add_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Build model first to get actual output shape
    print("\n[Model] Building detector...")
    model = build_ssd_detector_with_metadata(
        image_size=image_size,
        num_classes=num_classes,
        metadata_dim=metadata_dim,
        backbone_type=pretrained_model_type,
        freeze_backbone=True,  # Start frozen
        num_anchors=num_anchors,
    )
    
    # Get actual output shape from model (avoid test forward pass that might store tensors)
    # The model's output shape is available without a forward pass
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0] if output_shape else None
    print(f"[Model] Output shape: {output_shape}")
    print(f"[Model] Grid size: {grid_h}x{grid_w}")
    
    model.summary()
    
    # Create anchor encoder
    encoder = make_anchor_encoder_with_metadata(num_classes, grid_size, num_anchors=num_anchors)
    
    train_ds = train_ds_with_meta.map(encoder, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds_with_meta.map(encoder, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # Compile
    loss_fn = DetectionLossFocal(
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        positive_weight=positive_weight,
        bbox_loss_weight=bbox_loss_weight,
        num_anchors=num_anchors,
        use_anchors=True,
        objectness_label_smoothing=objectness_label_smoothing,
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
    
    # Custom ModelCheckpoint that handles JSON serialization errors
    class SafeModelCheckpoint(keras.callbacks.ModelCheckpoint):
        """ModelCheckpoint that handles JSON serialization errors gracefully."""
        def on_epoch_end(self, epoch, logs=None):
            try:
                super().on_epoch_end(epoch, logs)
            except (TypeError, ValueError) as e:
                if "serialize" in str(e).lower() or "json" in str(e).lower() or "EagerTensor" in str(e):
                    # Fallback: save weights only
                    print(f"\n[Warning] Could not save full model due to serialization error: {e}")
                    print(f"[Warning] Saving weights only instead...")
                    weights_path = str(self.filepath).replace('.keras', '_weights.h5')
                    try:
                        self.model.save_weights(weights_path)
                        print(f"[Warning] Saved weights to {weights_path}")
                    except Exception as e2:
                        print(f"[Error] Could not save weights either: {e2}")
                else:
                    raise
    
    # Create callbacks for Stage 1
    callbacks_stage1 = [
        SafeModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
            save_format="keras",  # Explicitly use Keras format
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
        SafeModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
            save_format="keras",  # Explicitly use Keras format
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
                "cspdarknet" in layer_name_lower or
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
    
    # Save model explicitly with Keras format to avoid serialization issues
    try:
        model.save(str(last_model_path), save_format="keras")
        print(f"\nSaved model to {last_model_path}")
    except Exception as e:
        print(f"\nWarning: Could not save model in Keras format: {e}")
        # Fallback: try saving weights only
        model.save_weights(str(last_model_path).replace('.keras', '_weights.h5'))
        print(f"Saved model weights instead")
    print(f"Best model (by val_loss) at {best_model_path}")


if __name__ == "__main__":
    main()

