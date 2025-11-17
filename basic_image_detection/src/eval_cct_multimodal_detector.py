"""
Evaluate multimodal CCT detector.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
import numpy as np

from coco_tfds_pipeline import make_coco_dataset
from cct_pipeline import make_cct_dataset, load_cct_annotations
from cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from train_simple_detector import infer_grid_size
from train_cct_multimodal_detector import (
    DetectionLossFocal,
    objectness_accuracy,
    build_multimodal_detector,
)


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [ymin, xmin, ymax, xmax] format."""
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    
    # Intersection
    inter_ymin = max(ymin1, ymin2)
    inter_xmin = max(xmin1, xmin2)
    inter_ymax = min(ymax1, ymax2)
    inter_xmax = min(xmax1, xmax2)
    
    if inter_ymax <= inter_ymin or inter_xmax <= inter_xmin:
        return 0.0
    
    inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
    box1_area = (ymax1 - ymin1) * (xmax1 - xmin1)
    box2_area = (ymax2 - ymin2) * (xmax2 - xmin2)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def nms(boxes, scores, iou_threshold=0.5, max_output_size=50):
    """
    Non-maximum suppression to remove overlapping boxes.
    
    Args:
        boxes: List of dicts with "bbox" and "score"
        scores: List of scores (or use score from boxes)
        iou_threshold: IoU threshold for suppression
        max_output_size: Maximum number of boxes to keep
    
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score (descending)
    indices = sorted(range(len(boxes)), key=lambda i: boxes[i]["score"], reverse=True)
    
    keep = []
    while len(indices) > 0 and len(keep) < max_output_size:
        # Take highest scoring box
        current_idx = indices[0]
        keep.append(current_idx)
        indices = indices[1:]
        
        # Remove boxes with high IoU
        current_box = boxes[current_idx]["bbox"]
        remaining_indices = []
        for idx in indices:
            other_box = boxes[idx]["bbox"]
            iou = compute_iou(current_box, other_box)
            if iou < iou_threshold:
                remaining_indices.append(idx)
        indices = remaining_indices
    
    return keep


def decode_predictions(grid_pred, num_classes, threshold=0.5, nms_iou=0.5, max_boxes=20, min_box_size=0.01):
    """
    Decode grid predictions to bounding boxes with NMS.
    
    Input:  grid_pred: [S, S, 5 + C]
    Output: list of dicts with bbox, class_id, score
    """
    S = grid_pred.shape[0]
    boxes = []
    
    for gy in range(S):
        for gx in range(S):
            cell = grid_pred[gy, gx]
            
            obj = cell[0]
            if obj < threshold:
                continue
            
            cx, cy, w, h = cell[1:5]
            
            # Filter out very small boxes (likely false positives)
            if w < min_box_size or h < min_box_size:
                continue
            
            # Convert from center format
            xmin = cx - w / 2.0
            ymin = cy - h / 2.0
            xmax = cx + w / 2.0
            ymax = cy + h / 2.0
            
            # Clip to [0, 1]
            xmin = np.clip(xmin, 0, 1)
            ymin = np.clip(ymin, 0, 1)
            xmax = np.clip(xmax, 0, 1)
            ymax = np.clip(ymax, 0, 1)
            
            # Skip if box is too small after clipping
            if (xmax - xmin) < min_box_size or (ymax - ymin) < min_box_size:
                continue
            
            # Class
            class_probs = cell[5:]
            class_id = int(np.argmax(class_probs))
            score = float(obj * class_probs[class_id])  # Combine objectness and class confidence
            
            boxes.append({
                "bbox": [ymin, xmin, ymax, xmax],
                "class_id": class_id,
                "score": score,
            })
    
    # Apply NMS
    if len(boxes) > 0:
        keep_indices = nms(boxes, None, iou_threshold=nms_iou, max_output_size=max_boxes)
        boxes = [boxes[i] for i in keep_indices]
    
    return boxes


def draw_boxes(image, gt_boxes, pred_boxes, class_names):
    """Draw ground truth and predicted boxes on image."""
    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.imshow(image)
    h, w, _ = image.shape
    
    # Ground truth (green)
    for (ymin, xmin, ymax, xmax) in gt_boxes:
        rect = patches.Rectangle(
            (xmin*w, ymin*h),
            (xmax-xmin)*w,
            (ymax-ymin)*h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
    
    # Predictions (red)
    for p in pred_boxes:
        (ymin, xmin, ymax, xmax) = p["bbox"]
        rect = patches.Rectangle(
            (xmin*w, ymin*h),
            (xmax-xmin)*w,
            (ymax-ymin)*h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            xmin*w,
            ymin*h,
            f"{class_names[p['class_id']]} ({p['score']:.2f})",
            color="red",
            fontsize=8,
            bbox=dict(facecolor="yellow", alpha=0.5),
        )
    
    ax.axis("off")
    plt.show()


def load_config():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "coco_multilabel_config.json"
    with open(config_path, "r") as f:
        return json.load(f), project_root


def main():
    config, project_root = load_config()
    
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    val_split = config["val_split"]
    
    model_name = config.get("detector_model_name", "cct_multimodal_detector")
    models_dir = project_root / config["models_dir"]
    model_path = models_dir / f"{model_name}_best.keras"
    
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 5.0)
    
    print("Loading model:", model_path)
    
    # Use the class-based loss for proper deserialization
    loss_fn = DetectionLossFocal(
        focal_gamma=focal_gamma, 
        focal_alpha=focal_alpha,
        positive_weight=positive_weight
    )
    
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "DetectionLossFocal": DetectionLossFocal,
            "detection_loss_focal": loss_fn,  # Also register as function name in case it was saved that way
            "objectness_accuracy": objectness_accuracy,
        }
    )
    
    model.summary()
    
    # Build dataset
    dataset_name = config.get("dataset", "cct")
    metadata_dim = 5
    
    if dataset_name == "cct":
        use_tfrecords = config.get("cct_use_tfrecords", True)
        cct_tfrecords_dir = config.get("cct_tfrecords_dir")
        
        if use_tfrecords and cct_tfrecords_dir:
            from pathlib import Path
            if Path(cct_tfrecords_dir).exists():
                print(f"[CCT] Using TFRecords from {cct_tfrecords_dir}")
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
            from cct_splits_utils import get_filelist_from_splits_or_config
            
            val_filelist = get_filelist_from_splits_or_config(config, "val", config["cct_annotations"])
            
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
    else:
        raise ValueError(f"Multimodal detector currently only supports CCT dataset")
    
    num_classes = val_info.features["objects"]["label"].num_classes
    class_names = val_info.features["objects"]["label"].names
    grid_size = infer_grid_size(image_size)
    
    # Add metadata and encode
    # Try to extract real metadata if using JSON pipeline
    cct_tfrecords_dir = config.get("cct_tfrecords_dir")
    use_tfrecords = config.get("cct_use_tfrecords", True) and cct_tfrecords_dir and Path(cct_tfrecords_dir).exists()
    
    if not use_tfrecords:
        # Load samples for metadata extraction
        from cct_pipeline import load_cct_annotations, extract_cct_metadata_features
        from cct_splits_utils import get_filelist_from_splits_or_config
        
        val_filelist = get_filelist_from_splits_or_config(config, "val", config["cct_annotations"])
        samples_val, _ = load_cct_annotations(
            metadata_path=config["cct_annotations"],
            bboxes_path=config["cct_bb_annotations"],
            images_root=config["cct_images_root"],
            filelist_path=val_filelist,
        )
        val_metadata_array = np.array([extract_cct_metadata_features(s) for s in samples_val], dtype=np.float32)
        
        # Unbatch, zip with metadata, then batch
        val_ds_unbatched = val_ds_raw.unbatch()
        val_metadata_ds = tf.data.Dataset.from_tensor_slices(val_metadata_array)
        val_ds_zipped = tf.data.Dataset.zip((val_ds_unbatched, val_metadata_ds))
        
        def combine_metadata(images_targets, metadata):
            images, targets = images_targets
            brightness = tf.reduce_mean(images, axis=[1, 2, 3])
            if len(brightness.shape) == 0:
                brightness = tf.expand_dims(brightness, 0)
            metadata_updated = tf.concat([
                metadata[:4] if len(metadata.shape) == 1 else metadata[:, :4],
                tf.expand_dims(brightness, -1)
            ], axis=-1)
            return (images, metadata_updated), targets
        
        val_ds_with_meta = val_ds_zipped.map(combine_metadata, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    else:
        # Use placeholder metadata for TFRecords
        def add_metadata_fn(images, targets):
            batch_size = tf.shape(images)[0]
            brightness = tf.reduce_mean(images, axis=[1, 2, 3])
            location_id = tf.zeros([batch_size], dtype=tf.float32)
            hour = tf.ones([batch_size], dtype=tf.float32) * 0.5
            day_of_week = tf.ones([batch_size], dtype=tf.float32) * 0.5
            month = tf.ones([batch_size], dtype=tf.float32) * 0.5
            metadata = tf.stack([location_id, hour, day_of_week, month, brightness], axis=1)
            return (images, metadata), targets
        
        val_ds_with_meta = val_ds_raw.map(add_metadata_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    from train_simple_detector import make_grid_encoder
    encoder = make_grid_encoder(num_classes, grid_size)
    
    def encode_with_metadata(inputs, targets):
        images, metadata = inputs
        images_encoded, grid_targets = encoder(images, targets)
        return (images_encoded, metadata), grid_targets
    
    val_ds = val_ds_raw.map(add_metadata_fn, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(encode_with_metadata, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # Numeric evaluation
    print("\n=== Running numeric evaluation ===")
    result = model.evaluate(val_ds, return_dict=True)
    print(result)
    
    # Visualization
    print("\n=== Visualizing predictions ===")
    
    for (inputs, grid_true) in val_ds.take(3):
        images, metadata = inputs
        preds = model([images, metadata], training=False).numpy()
        
        B = images.shape[0]
        
        for i in range(B):
            image = images[i].numpy()
            grid_t = grid_true[i].numpy()
            grid_p = preds[i]
            
            # Extract ground-truth boxes
            gt_boxes = []
            S = grid_size
            for gy in range(S):
                for gx in range(S):
                    cell = grid_t[gy, gx]
                    if cell[0] < 0.5:
                        continue
                    cx, cy, w, h = cell[1:5]
                    xmin = cx - w/2
                    ymin = cy - h/2
                    xmax = cx + w/2
                    ymax = cy + h/2
                    gt_boxes.append([ymin, xmin, ymax, xmax])
            
            pred_boxes = decode_predictions(grid_p, num_classes, threshold=0.5, nms_iou=0.5, max_boxes=10)
            
            print(f"\nImage {i+1}:")
            print(f"  GT boxes: {len(gt_boxes)}")
            print(f"  Pred boxes: {len(pred_boxes)}")
            
            draw_boxes(image, gt_boxes, pred_boxes, class_names)
        
        break


if __name__ == "__main__":
    main()

