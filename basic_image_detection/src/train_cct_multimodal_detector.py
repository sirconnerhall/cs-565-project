"""
Train multimodal detection model for CCT dataset.

Uses image + metadata (location, date, time) to improve detection accuracy.
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from coco_tfds_pipeline import make_coco_dataset
from cct_pipeline import make_cct_dataset, load_cct_annotations, extract_cct_metadata_features
from cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from train_simple_detector import infer_grid_size, make_grid_encoder
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
# Focal loss for objectness
# ----------------------------

def focal_loss_objectness(gamma=2.0, alpha=0.25):
    """
    Focal loss for objectness predictions to handle class imbalance.
    """
    def _loss(y_true, y_pred):
        # y_true, y_pred: [B, S, S, 1] (objectness)
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        
        # BCE
        ce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        
        # p_t
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        
        # alpha, modulating
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        loss = alpha_factor * modulating_factor * ce
        return tf.reduce_mean(loss)
    
    _loss.__name__ = "focal_loss_objectness"
    return _loss


# ----------------------------
# Detection loss with focal for objectness
# ----------------------------

class DetectionLossFocal(keras.losses.Loss):
    """
    Detection loss with focal loss for objectness.
    Inherits from keras.losses.Loss for proper serialization.
    
    For severe class imbalance (many empty images), use:
    - Higher focal_alpha (0.5-0.75) to weight positive examples more
    - Higher positive_weight (2.0-10.0) to further emphasize objects
    - Higher focal_gamma (2.0-3.0) to focus on hard negatives
    """
    def __init__(self, focal_gamma=2.0, focal_alpha=0.5, positive_weight=5.0, name="detection_loss_focal", **kwargs):
        super().__init__(name=name, **kwargs)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.positive_weight = positive_weight  # Additional weight for positive examples
    
    def call(self, y_true, y_pred):
        """Compute the loss."""
        return self._compute_loss(y_true, y_pred)
    
    def _compute_loss(self, y_true, y_pred):
        """Compute the actual loss."""
        # Objectness with focal loss + positive weighting
        obj_true = y_true[..., 0:1]   # [B, S, S, 1]
        obj_pred = y_pred[..., 0:1]   # [B, S, S, 1]
        
        eps = 1e-7
        obj_pred_clipped = tf.clip_by_value(obj_pred, eps, 1.0 - eps)
        ce = -(obj_true * tf.math.log(obj_pred_clipped) + 
               (1.0 - obj_true) * tf.math.log(1.0 - obj_pred_clipped))
        p_t = obj_true * obj_pred_clipped + (1.0 - obj_true) * (1.0 - obj_pred_clipped)
        alpha_factor = obj_true * self.focal_alpha + (1.0 - obj_true) * (1.0 - self.focal_alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.focal_gamma)
        
        # Apply additional positive weight to emphasize objects
        positive_mask = obj_true  # 1.0 where there's an object, 0.0 otherwise
        weight_factor = 1.0 + positive_mask * (self.positive_weight - 1.0)
        
        obj_loss = alpha_factor * modulating_factor * ce * weight_factor
        
        # Bboxes (only where there is an object)
        box_true = y_true[..., 1:5]   # [B, S, S, 4]
        box_pred = y_pred[..., 1:5]   # [B, S, S, 4]
        box_diff = box_true - box_pred
        box_sq = tf.square(box_diff)
        box_loss = tf.reduce_sum(box_sq, axis=-1, keepdims=True)  # [B, S, S, 1]
        box_loss = box_loss * obj_true  # Mask by objectness
        
        # Classes (only where there is an object)
        cls_true = y_true[..., 5:]   # [B, S, S, C]
        cls_pred = y_pred[..., 5:]   # [B, S, S, C]
        cls_bce = tf.keras.backend.binary_crossentropy(cls_true, cls_pred)
        cls_loss = tf.reduce_mean(cls_bce, axis=-1, keepdims=True)  # [B, S, S, 1]
        cls_loss = cls_loss * obj_true  # Mask by objectness
        
        # Sum all three terms
        total = obj_loss + box_loss + cls_loss
        return tf.reduce_mean(total)
    
    def get_config(self):
        return {
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "positive_weight": self.positive_weight,
        }


# ----------------------------
# Per-component loss metrics
# ----------------------------

def make_component_loss_metrics(loss_fn):
    """
    Create metrics that track individual loss components.
    
    Args:
        loss_fn: DetectionLossFocal instance
    
    Returns:
        List of metric functions for objectness, bbox, and class losses
    """
    def objectness_loss_metric(y_true, y_pred):
        """Compute objectness loss component."""
        obj_true = y_true[..., 0:1]
        obj_pred = y_pred[..., 0:1]
        
        eps = 1e-7
        obj_pred_clipped = tf.clip_by_value(obj_pred, eps, 1.0 - eps)
        ce = -(obj_true * tf.math.log(obj_pred_clipped) + 
               (1.0 - obj_true) * tf.math.log(1.0 - obj_pred_clipped))
        p_t = obj_true * obj_pred_clipped + (1.0 - obj_true) * (1.0 - obj_pred_clipped)
        alpha_factor = obj_true * loss_fn.focal_alpha + (1.0 - obj_true) * (1.0 - loss_fn.focal_alpha)
        modulating_factor = tf.pow(1.0 - p_t, loss_fn.focal_gamma)
        positive_mask = obj_true
        weight_factor = 1.0 + positive_mask * (loss_fn.positive_weight - 1.0)
        obj_loss = alpha_factor * modulating_factor * ce * weight_factor
        return tf.reduce_mean(obj_loss)
    
    def bbox_loss_metric(y_true, y_pred):
        """Compute bounding box loss component."""
        obj_true = y_true[..., 0:1]
        box_true = y_true[..., 1:5]
        box_pred = y_pred[..., 1:5]
        box_diff = box_true - box_pred
        box_sq = tf.square(box_diff)
        box_loss = tf.reduce_sum(box_sq, axis=-1, keepdims=True)
        box_loss = box_loss * obj_true  # Mask by objectness
        return tf.reduce_mean(box_loss)
    
    def class_loss_metric(y_true, y_pred):
        """Compute classification loss component."""
        obj_true = y_true[..., 0:1]
        cls_true = y_true[..., 5:]
        cls_pred = y_pred[..., 5:]
        cls_bce = tf.keras.backend.binary_crossentropy(cls_true, cls_pred)
        cls_loss = tf.reduce_mean(cls_bce, axis=-1, keepdims=True)
        cls_loss = cls_loss * obj_true  # Mask by objectness
        return tf.reduce_mean(cls_loss)
    
    objectness_loss_metric.__name__ = "objectness_loss"
    bbox_loss_metric.__name__ = "bbox_loss"
    class_loss_metric.__name__ = "class_loss"
    
    return [objectness_loss_metric, bbox_loss_metric, class_loss_metric]


def detection_loss_focal(y_true, y_pred, focal_gamma=2.0, focal_alpha=0.25):
    """
    Combined loss with focal loss for objectness:
      - objectness focal loss for all cells
      - bbox MSE for cells with an object
      - class BCE for cells with an object
    
    Shapes:
        y_true, y_pred: [B, S, S, 5 + C]
    """
    # Objectness with focal loss
    obj_true = y_true[..., 0:1]   # [B, S, S, 1]
    obj_pred = y_pred[..., 0:1]   # [B, S, S, 1]
    
    eps = 1e-7
    obj_pred_clipped = tf.clip_by_value(obj_pred, eps, 1.0 - eps)
    ce = -(obj_true * tf.math.log(obj_pred_clipped) + 
           (1.0 - obj_true) * tf.math.log(1.0 - obj_pred_clipped))
    p_t = obj_true * obj_pred_clipped + (1.0 - obj_true) * (1.0 - obj_pred_clipped)
    alpha_factor = obj_true * focal_alpha + (1.0 - obj_true) * (1.0 - focal_alpha)
    modulating_factor = tf.pow(1.0 - p_t, focal_gamma)
    obj_loss = alpha_factor * modulating_factor * ce
    
    # Bboxes (only where there is an object)
    box_true = y_true[..., 1:5]   # [B, S, S, 4]
    box_pred = y_pred[..., 1:5]   # [B, S, S, 4]
    box_diff = box_true - box_pred
    box_sq = tf.square(box_diff)
    box_loss = tf.reduce_sum(box_sq, axis=-1, keepdims=True)  # [B, S, S, 1]
    box_loss = box_loss * obj_true  # Mask by objectness
    
    # Classes (only where there is an object)
    cls_true = y_true[..., 5:]   # [B, S, S, C]
    cls_pred = y_pred[..., 5:]   # [B, S, S, C]
    cls_bce = tf.keras.backend.binary_crossentropy(cls_true, cls_pred)
    cls_loss = tf.reduce_mean(cls_bce, axis=-1, keepdims=True)  # [B, S, S, 1]
    cls_loss = cls_loss * obj_true  # Mask by objectness
    
    # Sum all three terms
    total = obj_loss + box_loss + cls_loss
    return tf.reduce_mean(total)


# ----------------------------
# Extract metadata from samples
# ----------------------------

def extract_metadata_batch(samples_batch):
    """
    Extract metadata features from a batch of samples.
    
    Args:
        samples_batch: List of sample dicts with location, date_captured
    
    Returns:
        [B, metadata_dim] numpy array
    """
    metadata_list = []
    for sample in samples_batch:
        metadata = extract_cct_metadata_features(sample)
        metadata_list.append(metadata)
    return np.array(metadata_list, dtype=np.float32)


def add_metadata_to_dataset(ds, samples, metadata_dim=5):
    """
    Add metadata to dataset by mapping samples to metadata features.
    
    Args:
        ds: Dataset yielding (images, targets)
        samples: List of sample dicts
        metadata_dim: Dimension of metadata vector
    
    Returns:
        Dataset yielding ((images, metadata), targets)
    """
    # Create mapping from index to metadata
    metadata_array = np.array([extract_cct_metadata_features(s) for s in samples], dtype=np.float32)
    
    def _add_metadata(images, targets):
        # Get batch size
        batch_size = tf.shape(images)[0]
        
        # For simplicity, we'll use a fixed metadata vector per batch
        # In practice, you'd need to track which samples are in each batch
        # This is a simplified version - for production, use dataset.enumerate() or similar
        metadata_batch = tf.constant(metadata_array[:batch_size], dtype=tf.float32)
        if tf.shape(metadata_batch)[0] < batch_size:
            # Pad if needed
            padding = tf.zeros((batch_size - tf.shape(metadata_batch)[0], metadata_dim), dtype=tf.float32)
            metadata_batch = tf.concat([metadata_batch, padding], axis=0)
        
        return (images, metadata_batch), targets
    
    return ds.map(_add_metadata, num_parallel_calls=tf.data.AUTOTUNE)


# ----------------------------
# Multimodal detection model
# ----------------------------

def build_multimodal_detector(image_size, num_classes, metadata_dim=5):
    """
    Build multimodal detection model.
    
    Image branch: MobileNetV2 → feature map (spatial dimensions preserved)
    Metadata branch: Dense layers
    Fusion: Metadata features injected into image features via FiLM-like conditioning
    
    Output: [B, S, S, 5 + C] grid predictions
    """
    h, w = image_size
    grid_size = infer_grid_size(image_size)
    
    # Image input
    image_input = keras.Input(shape=(h, w, 3), name="image")
    
    # Image branch (keep spatial dimensions)
    base_model = keras.applications.MobileNetV2(
        input_shape=(h, w, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Start frozen
    
    x_img = keras.applications.mobilenet_v2.preprocess_input(image_input)
    x_img = base_model(x_img, training=False)  # [B, H', W', C']
    
    # Metadata input
    metadata_input = keras.Input(shape=(metadata_dim,), name="metadata")
    
    # Metadata branch
    x_meta = layers.Dense(64, activation="relu")(metadata_input)
    x_meta = layers.Dense(32, activation="relu")(x_meta)
    x_meta = layers.Dense(16, activation="relu")(x_meta)  # [B, 16]
    
    # Expand metadata to spatial dimensions for FiLM-like conditioning
    # Get spatial dimensions from image features
    _, h_feat, w_feat, c_feat = x_img.shape
    x_meta_expanded = layers.Dense(c_feat * 2)(x_meta)  # [B, C' * 2]
    x_meta_expanded = tf.reshape(x_meta_expanded, [-1, 1, 1, c_feat * 2])  # [B, 1, 1, C' * 2]
    x_meta_expanded = tf.tile(x_meta_expanded, [1, h_feat, w_feat, 1])  # [B, H', W', C' * 2]
    
    # Split into scale and shift (FiLM conditioning)
    scale = x_meta_expanded[..., :c_feat]  # [B, H', W', C']
    shift = x_meta_expanded[..., c_feat:]  # [B, H', W', C']
    
    # Apply FiLM conditioning
    x_img = x_img * (1.0 + scale) + shift
    
    # Detection head
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x_img)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    
    # Output: [B, S, S, 5 + C]
    outputs = layers.Conv2D(
        5 + num_classes,
        1,
        padding="same",
        activation="sigmoid",
        name="grid_output",
    )(x)
    
    model = keras.Model(
        inputs=[image_input, metadata_input],
        outputs=outputs,
        name="cct_multimodal_detector",
    )
    return model


# ----------------------------
# Objectness accuracy metric
# ----------------------------

def objectness_accuracy(y_true, y_pred):
    obj_true = y_true[..., 0]
    obj_pred = y_pred[..., 0]
    obj_pred_label = tf.cast(obj_pred > 0.5, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(obj_true, obj_pred_label), tf.float32))


# ----------------------------
# Diagnostic metrics
# ----------------------------

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
        
        # Per-component losses
        if 'objectness_loss' in logs:
            print(f"  Loss Components:")
            print(f"    Objectness: {logs.get('objectness_loss', 'N/A'):.4f} (val: {logs.get('val_objectness_loss', 'N/A'):.4f})")
            print(f"    BBox: {logs.get('bbox_loss', 'N/A'):.4f} (val: {logs.get('val_bbox_loss', 'N/A'):.4f})")
            print(f"    Class: {logs.get('class_loss', 'N/A'):.4f} (val: {logs.get('val_class_loss', 'N/A'):.4f})")
        
        # Sample validation batch and compute detailed stats
        if self.val_dataset is not None and self.num_classes is not None and self.grid_size is not None:
            try:
                # Get one batch from validation set
                val_batch = next(iter(self.val_dataset))
                if isinstance(val_batch, tuple) and len(val_batch) == 2:
                    inputs, targets = val_batch
                    if isinstance(inputs, tuple):
                        images, metadata = inputs
                    else:
                        images = inputs
                        metadata = None
                else:
                    return
                
                # Get predictions
                if metadata is not None:
                    predictions = self.model([images, metadata], training=False)
                else:
                    predictions = self.model(images, training=False)
                
                # Convert to numpy for analysis
                pred_np = predictions.numpy()  # [B, S, S, 5+C]
                target_np = targets.numpy()    # [B, S, S, 5+C]
                
                batch_size = pred_np.shape[0]
                S = pred_np.shape[1]
                
                # Cell-level objectness stats
                obj_true = target_np[..., 0]  # [B, S, S]
                obj_pred = pred_np[..., 0]   # [B, S, S]
                obj_pred_binary = (obj_pred >= self.threshold).astype(np.float32)
                
                # True positives, false positives, false negatives
                tp = np.sum((obj_true == 1) & (obj_pred_binary == 1))
                fp = np.sum((obj_true == 0) & (obj_pred_binary == 1))
                fn = np.sum((obj_true == 1) & (obj_pred_binary == 0))
                tn = np.sum((obj_true == 0) & (obj_pred_binary == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                # Average objectness scores
                obj_score_with_gt = np.mean(obj_pred[obj_true == 1]) if np.any(obj_true == 1) else 0.0
                obj_score_without_gt = np.mean(obj_pred[obj_true == 0]) if np.any(obj_true == 0) else 0.0
                
                # Count boxes per image (using simple decoding without NMS for speed)
                pred_boxes_per_image = []
                gt_boxes_per_image = []
                images_with_pred = 0
                images_with_gt = 0
                images_with_gt_and_pred = 0
                
                for b in range(batch_size):
                    # Count GT boxes
                    gt_obj = obj_true[b]
                    gt_count = int(np.sum(gt_obj > 0.5))
                    gt_boxes_per_image.append(gt_count)
                    if gt_count > 0:
                        images_with_gt += 1
                    
                    # Count predicted boxes (simple threshold, no NMS)
                    pred_obj = obj_pred[b]
                    pred_count = int(np.sum(pred_obj >= self.threshold))
                    pred_boxes_per_image.append(pred_count)
                    if pred_count > 0:
                        images_with_pred += 1
                        if gt_count > 0:
                            images_with_gt_and_pred += 1
                
                avg_pred_boxes = np.mean(pred_boxes_per_image)
                avg_gt_boxes = np.mean(gt_boxes_per_image)
                pct_with_pred = (images_with_pred / batch_size) * 100.0
                pct_with_gt = (images_with_gt / batch_size) * 100.0
                pct_gt_detected = (images_with_gt_and_pred / images_with_gt * 100.0) if images_with_gt > 0 else 0.0
                
                print(f"\n  Validation Batch Statistics (threshold={self.threshold}):")
                print(f"    Cell-level Objectness:")
                print(f"      TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
                print(f"      Precision: {precision:.4f}, Recall: {recall:.4f}")
                print(f"      Avg objectness (with GT): {obj_score_with_gt:.4f}")
                print(f"      Avg objectness (without GT): {obj_score_without_gt:.4f}")
                print(f"    Box-level Statistics:")
                print(f"      Avg predicted boxes/image: {avg_pred_boxes:.2f}")
                print(f"      Avg GT boxes/image: {avg_gt_boxes:.2f}")
                print(f"      Images with ≥1 pred: {images_with_pred}/{batch_size} ({pct_with_pred:.1f}%)")
                print(f"      Images with ≥1 GT: {images_with_gt}/{batch_size} ({pct_with_gt:.1f}%)")
                print(f"      GT images detected: {images_with_gt_and_pred}/{images_with_gt} ({pct_gt_detected:.1f}%)")
                
            except Exception as e:
                print(f"  Warning: Could not compute detailed stats: {e}")
        
        print(f"{'='*60}")


# ----------------------------
# Main training loop
# ----------------------------

def main():
    config, project_root = load_config("coco_multilabel_config.json")
    
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    train_split = config["train_split"]
    val_split = config["val_split"]
    learning_rate = config["learning_rate"]
    models_dir = project_root / config["models_dir"]
    
    model_name = config.get("detector_model_name", "cct_multimodal_detector")
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)  # Increased default for imbalanced data
    positive_weight = config.get("positive_weight", 5.0)  # Additional weight for positive examples
    filter_empty_images = config.get("filter_empty_images", False)  # Option to filter empty images
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Project root:", project_root)
    print("Image size: ", image_size)
    print("Focal loss: gamma={}, alpha={}".format(focal_gamma, focal_alpha))
    
    dataset_name = config.get("dataset", "cct")
    metadata_dim = 5  # location_id, hour, day_of_week, month, brightness
    
    if dataset_name == "cct":
        # Load samples for metadata extraction
        from cct_splits_utils import get_filelist_from_splits_or_config
        
        train_filelist = get_filelist_from_splits_or_config(config, "train", config["cct_annotations"])
        val_filelist = get_filelist_from_splits_or_config(config, "val", config["cct_annotations"])
        
        samples_train, _ = load_cct_annotations(
            metadata_path=config["cct_annotations"],
            bboxes_path=config["cct_bb_annotations"],
            images_root=config["cct_images_root"],
            filelist_path=train_filelist,
            filter_empty=filter_empty_images,
        )
        samples_val, _ = load_cct_annotations(
            metadata_path=config["cct_annotations"],
            bboxes_path=config["cct_bb_annotations"],
            images_root=config["cct_images_root"],
            filelist_path=val_filelist,
            filter_empty=filter_empty_images,
        )
        
        if filter_empty_images:
            print(f"[Dataset] Filtered empty images: {len(samples_train)} train, {len(samples_val)} val samples with objects")
        
        # Check for TFRecords
        use_tfrecords = config.get("cct_use_tfrecords", True)
        cct_tfrecords_dir = config.get("cct_tfrecords_dir")
        
        if use_tfrecords and cct_tfrecords_dir:
            from pathlib import Path
            if Path(cct_tfrecords_dir).exists():
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
                use_tfrecords = False
        
        if not use_tfrecords:
            print(f"[CCT] Using JSON pipeline")
            # train_filelist and val_filelist already set above
            if train_filelist is None:
                train_filelist = get_filelist_from_splits_or_config(config, "train", config["cct_annotations"])
            if val_filelist is None:
                val_filelist = get_filelist_from_splits_or_config(config, "val", config["cct_annotations"])
            
            train_ds_raw, train_info = make_cct_dataset(
                images_root=config["cct_images_root"],
                metadata_path=config["cct_annotations"],
                bboxes_path=config["cct_bb_annotations"],
                filelist_path=train_filelist,
                split="train",
                batch_size=batch_size,
                image_size=image_size,
                filter_empty=filter_empty_images,
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
                filter_empty=filter_empty_images,
            )
    else:
        raise ValueError(f"Multimodal detector currently only supports CCT dataset")
    
    num_classes = train_info.features["objects"]["label"].num_classes
    print(f"{dataset_name.upper()} num classes: {num_classes}")
    
    grid_size = infer_grid_size(image_size)
    print("Grid size (SxS):", grid_size, "x", grid_size)
    
    # Pre-extract metadata from samples
    # Note: This only works with JSON pipeline, not TFRecords
    # For TFRecords, metadata would need to be stored in the records
    if not use_tfrecords:
        print("[Metadata] Extracting metadata from samples...")
        train_metadata_array = np.array([extract_cct_metadata_features(s) for s in samples_train], dtype=np.float32)
        val_metadata_array = np.array([extract_cct_metadata_features(s) for s in samples_val], dtype=np.float32)
        print(f"[Metadata] Extracted metadata for {len(train_metadata_array)} train and {len(val_metadata_array)} val samples")
        
        # Create metadata datasets - need to unbatch image dataset first to align properly
        # Then we'll batch both together
        train_ds_unbatched = train_ds_raw.unbatch()
        val_ds_unbatched = val_ds_raw.unbatch()
        
        train_metadata_ds = tf.data.Dataset.from_tensor_slices(train_metadata_array)
        val_metadata_ds = tf.data.Dataset.from_tensor_slices(val_metadata_array)
        
        # Zip before batching to keep alignment
        train_ds_zipped = tf.data.Dataset.zip((train_ds_unbatched, train_metadata_ds))
        val_ds_zipped = tf.data.Dataset.zip((val_ds_unbatched, val_metadata_ds))
    else:
        # For TFRecords, use placeholder metadata (location/time would need to be in TFRecords)
        print("[Metadata] Using placeholder metadata for TFRecords (location/time not available)")
        def add_placeholder_metadata(images, targets):
            batch_size = tf.shape(images)[0]
            brightness = tf.reduce_mean(images, axis=[1, 2, 3])
            location_id = tf.zeros([batch_size], dtype=tf.float32)
            hour = tf.ones([batch_size], dtype=tf.float32) * 0.5
            day_of_week = tf.ones([batch_size], dtype=tf.float32) * 0.5
            month = tf.ones([batch_size], dtype=tf.float32) * 0.5
            metadata = tf.stack([location_id, hour, day_of_week, month, brightness], axis=1)
            return (images, metadata), targets
        
        train_ds_zipped = train_ds_raw.map(add_placeholder_metadata, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds_zipped = val_ds_raw.map(add_placeholder_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    
    if not use_tfrecords:
        def combine_metadata(images_targets, metadata):
            images, targets = images_targets
            # Compute brightness from images (override the placeholder)
            brightness = tf.reduce_mean(images, axis=[1, 2, 3])  # scalar or [B]
            
            # Handle both batched and unbatched cases
            if len(brightness.shape) == 0:
                brightness = tf.expand_dims(brightness, 0)
            
            # Update brightness in metadata (last element)
            metadata_updated = tf.concat([
                metadata[:4] if len(metadata.shape) == 1 else metadata[:, :4],  # location_id, hour, day_of_week, month
                tf.expand_dims(brightness, -1)  # brightness
            ], axis=-1)
            
            return (images, metadata_updated), targets
        
        train_ds_with_meta = train_ds_zipped.map(combine_metadata, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds_with_meta = val_ds_zipped.map(combine_metadata, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch after combining, shuffle training data
        train_ds_with_meta = train_ds_with_meta.shuffle(1024).batch(batch_size)
        val_ds_with_meta = val_ds_with_meta.batch(batch_size)
    else:
        # Already has metadata added
        train_ds_with_meta = train_ds_zipped
        val_ds_with_meta = val_ds_zipped
    
    # Encode to grid format
    encoder = make_grid_encoder(num_classes, grid_size)
    
    def encode_with_metadata(inputs, targets):
        images, metadata = inputs
        images_encoded, grid_targets = encoder(images, targets)
        return (images_encoded, metadata), grid_targets
    
    train_ds = train_ds_with_meta.map(encode_with_metadata, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds_with_meta.map(encode_with_metadata, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    model = build_multimodal_detector(image_size, num_classes, metadata_dim)
    model.summary()
    
    # Compile with focal loss
    # Use the class-based loss for proper serialization
    # For severe imbalance, use higher alpha and positive_weight
    loss_fn = DetectionLossFocal(
        focal_gamma=focal_gamma, 
        focal_alpha=focal_alpha,
        positive_weight=positive_weight
    )
    print(f"[Loss] Using focal loss: gamma={focal_gamma}, alpha={focal_alpha}, positive_weight={positive_weight}")
    
    # Create per-component loss metrics
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
            patience=5,  # Increased patience for small datasets
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,  # Increased patience
            mode="min",
            verbose=1,
        ),
        PredictionStats(threshold=0.5, val_dataset=val_ds, num_classes=num_classes, grid_size=grid_size),
    ]
    
    print("\n=== Training multimodal detector ===")
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

