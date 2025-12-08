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
from ..utils.anchor_encoder import make_anchor_encoder
from .build_detector import build_detector


def load_config(config_name="config.json"):
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / config_name
    with open(config_path, "r") as f:
        config = json.load(f)
    return config, project_root


def infer_grid_size(image_size):
    """Infer grid size from CSPDarkNet backbone stride (32x downsampling)."""
    h, w = image_size
    # CSPDarkNet downsamples by 32x
    grid_h = h // 32
    grid_w = w // 32
    return grid_h, grid_w


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
    objectness_label_smoothing = config.get("objectness_label_smoothing", 0.1)
    filter_empty_images = config.get("filter_empty_images", False)
    use_focal_loss = config.get("use_focal_loss", True)

    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training SSNM Detector (Single Stage, No Metadata)")
    print("=" * 60)
    print(f"Model name: {model_name}")
    print(f"Image size: {image_size}")
    print(f"Backbone: Frozen (single-stage transfer learning)")
    print(f"Backbone type: {config.get('pretrained_model_type', 'efficientnet_b0')}")

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
        
        print(f"[CCT] Using TFRecords from {cct_tfrecords_dir}")
        print(f"[CCT] Train split: {train_split_base}, Val split: {val_split_base}")
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
                    print(f"[CCT] Limiting train split to {percentage}% ({take_size} of {total_size} samples)")
                    train_ds_raw = train_ds_raw.take(take_size)
                else:
                    print(f"[CCT] Warning: Could not determine dataset size for percentage filtering")
        
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
                    print(f"[CCT] Limiting val split to {percentage}% ({take_size} of {total_size} samples)")
                    val_ds_raw = val_ds_raw.take(take_size)
                else:
                    print(f"[CCT] Warning: Could not determine dataset size for percentage filtering")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_classes = train_info.features["objects"]["label"].num_classes
    print(f"{dataset_name.upper()} num classes: {num_classes}")

    # Get grid size and anchor configuration
    grid_h, grid_w = infer_grid_size(image_size)
    grid_size = (grid_h, grid_w)
    num_anchors = config.get("num_anchors", 3)
    pretrained_model_type = config.get("pretrained_model_type", "efficientnet_b0")
    print(f"Grid size: {grid_h}x{grid_w}, num_anchors: {num_anchors}")
    print(f"Backbone type: {pretrained_model_type}")

    AUTOTUNE = tf.data.AUTOTUNE
    encoder = make_anchor_encoder(num_classes, grid_size, num_anchors=num_anchors)

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
    model = build_detector(
        image_size=image_size,
        num_classes=num_classes,
        num_anchors=num_anchors,
        backbone_type=pretrained_model_type,
    )
    model.summary()

    # Compile
    if use_focal_loss:
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

    callbacks = [
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
    

    print("\n=== Training SSNM Detector ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # Save model explicitly with Keras format to avoid serialization issues
    try:
        model.save(str(last_model_path), save_format="keras")
        print(f"\nSaved last model to {last_model_path}")
    except Exception as e:
        print(f"\nWarning: Could not save model in Keras format: {e}")
        # Fallback: try saving weights only
        model.save_weights(str(last_model_path).replace('.keras', '_weights.h5'))
        print(f"Saved model weights instead")
    print(f"Best model (by val_loss) at {best_model_path}")


if __name__ == "__main__":
    main()

