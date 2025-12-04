import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers  # works with tf.keras in TF 2.10

from ..pipelines.coco_tfds_pipeline import make_coco_dataset  # existing loader
from ..pipelines.cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from ..pipelines.coco_multilabel_utils import run_extended_sanity_checks

# ----------------------------
# Config helpers
# ----------------------------

def load_config(config_name="config.json"):
    project_root = Path(__file__).resolve().parents[2]
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
    - Higher bbox_loss_weight (5.0-20.0) to emphasize accurate box regression
    """
    def __init__(self, focal_gamma=2.0, focal_alpha=0.5, positive_weight=5.0, bbox_loss_weight=10.0, name="detection_loss_focal", **kwargs):
        super().__init__(name=name, **kwargs)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.positive_weight = positive_weight  # Additional weight for positive examples
        self.bbox_loss_weight = bbox_loss_weight  # Weight for bbox regression loss
    
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
        box_true = y_true[..., 1:5]   # [B, S, S, 4] - [cx, cy, w, h]
        box_pred = y_pred[..., 1:5]   # [B, S, S, 4] - [cx, cy, w, h]
        
        # Convert to corner format for IoU computation
        # True boxes
        cx_true, cy_true, w_true, h_true = tf.split(box_true, 4, axis=-1)
        xmin_true = cx_true - w_true / 2.0
        ymin_true = cy_true - h_true / 2.0
        xmax_true = cx_true + w_true / 2.0
        ymax_true = cy_true + h_true / 2.0
        
        # Predicted boxes
        cx_pred, cy_pred, w_pred, h_pred = tf.split(box_pred, 4, axis=-1)
        xmin_pred = cx_pred - w_pred / 2.0
        ymin_pred = cy_pred - h_pred / 2.0
        xmax_pred = cx_pred + w_pred / 2.0
        ymax_pred = cy_pred + h_pred / 2.0
        
        # Clip to [0, 1]
        xmin_true = tf.clip_by_value(xmin_true, 0.0, 1.0)
        ymin_true = tf.clip_by_value(ymin_true, 0.0, 1.0)
        xmax_true = tf.clip_by_value(xmax_true, 0.0, 1.0)
        ymax_true = tf.clip_by_value(ymax_true, 0.0, 1.0)
        xmin_pred = tf.clip_by_value(xmin_pred, 0.0, 1.0)
        ymin_pred = tf.clip_by_value(ymin_pred, 0.0, 1.0)
        xmax_pred = tf.clip_by_value(xmax_pred, 0.0, 1.0)
        ymax_pred = tf.clip_by_value(ymax_pred, 0.0, 1.0)
        
        # Compute IoU
        inter_xmin = tf.maximum(xmin_true, xmin_pred)
        inter_ymin = tf.maximum(ymin_true, ymin_pred)
        inter_xmax = tf.minimum(xmax_true, xmax_pred)
        inter_ymax = tf.minimum(ymax_true, ymax_pred)
        
        inter_w = tf.maximum(0.0, inter_xmax - inter_xmin)
        inter_h = tf.maximum(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        
        area_true = (xmax_true - xmin_true) * (ymax_true - ymin_true)
        area_pred = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
        union_area = area_true + area_pred - inter_area
        
        # IoU with small epsilon to avoid division by zero
        eps = 1e-7
        iou = inter_area / (union_area + eps)
        
        # Use 1 - IoU as loss (higher IoU = lower loss)
        box_loss = (1.0 - iou)  # [B, S, S, 1]
        box_loss = box_loss * obj_true  # Mask by objectness
        box_loss = box_loss * self.bbox_loss_weight  # Apply bbox loss weight
        
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
            "bbox_loss_weight": self.bbox_loss_weight,
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


# Legacy function for backward compatibility
def detection_loss(y_true, y_pred):
    """
    Legacy simple detection loss (BCE for objectness).
    Use DetectionLossFocal for better performance on imbalanced data.
    """
    # Objectness
    obj_true = y_true[..., 0:1]   # [B, S, S, 1]
    obj_pred = y_pred[..., 0:1]   # [B, S, S, 1]
    obj_loss = tf.keras.backend.binary_crossentropy(obj_true, obj_pred)

    # Bboxes (only where there is an object)
    box_true = y_true[..., 1:5]   # [B, S, S, 4]
    box_pred = y_pred[..., 1:5]   # [B, S, S, 4]
    box_diff = box_true - box_pred
    box_sq = tf.square(box_diff)
    box_loss = tf.reduce_sum(box_sq, axis=-1, keepdims=True)
    box_loss = box_loss * obj_true

    # Classes (only where there is an object)
    cls_true = y_true[..., 5:]   # [B, S, S, C]
    cls_pred = y_pred[..., 5:]   # [B, S, S, C]
    cls_bce = tf.keras.backend.binary_crossentropy(cls_true, cls_pred)
    cls_loss = tf.reduce_mean(cls_bce, axis=-1, keepdims=True)
    cls_loss = cls_loss * obj_true

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
                    images, targets = val_batch
                else:
                    return
                
                # Get predictions
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
    # Load config (reuse your existing config.json)
    config, project_root = load_config("config.json")

    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    train_split = config["train_split"]
    val_split = config["val_split"]
    learning_rate = config["learning_rate"]
    models_dir = project_root / config["models_dir"]

    # Optional: detector-specific model name in config; else fallback
    model_name = config.get("detector_model_name", "coco_simple_detector")
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)  # Increased default for imbalanced data
    positive_weight = config.get("positive_weight", 5.0)  # Additional weight for positive examples
    bbox_loss_weight = config.get("bbox_loss_weight", 10.0)  # Weight for bbox regression loss
    filter_empty_images = config.get("filter_empty_images", False)  # Option to filter empty images
    use_focal_loss = config.get("use_focal_loss", True)  # Use focal loss by default

    models_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", project_root)
    print("Train split:", train_split)
    print("Val split:  ", val_split)
    print("Image size: ", image_size)
    if use_focal_loss:
        print(f"Focal loss: gamma={focal_gamma}, alpha={focal_alpha}, positive_weight={positive_weight}")
    if filter_empty_images:
        print("Filtering empty images (images with no bboxes)")

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
        # Load from TFRecords
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

    # Compile with appropriate loss
    if use_focal_loss:
        loss_fn = DetectionLossFocal(
            focal_gamma=focal_gamma, 
            focal_alpha=focal_alpha,
            positive_weight=positive_weight,
            bbox_loss_weight=bbox_loss_weight
        )
        print(f"[Loss] Using focal loss: gamma={focal_gamma}, alpha={focal_alpha}, positive_weight={positive_weight}, bbox_weight={bbox_loss_weight}")
        # Create per-component loss metrics
        component_metrics = make_component_loss_metrics(loss_fn)
        metrics = [objectness_accuracy] + component_metrics
    else:
        loss_fn = detection_loss
        print("[Loss] Using simple BCE loss (legacy)")
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
