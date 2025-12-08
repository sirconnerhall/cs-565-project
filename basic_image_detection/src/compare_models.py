"""
Compare performance of all four detector models:
- SSNM: Single Stage, No Metadata
- SSM: Single Stage, with Metadata
- TSNM: Two Stage, No Metadata
- TSM: Two Stage, with Metadata
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .pipelines.cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from .utils.detection_utils import (
    DetectionLossFocal,
    make_component_loss_metrics,
    objectness_accuracy,
    decode_predictions_grid,
    decode_predictions_anchors,
    compute_iou,
    compute_map,
)
from .utils.film_layer import FiLMLayer

# Import model builders
from .SSNM.build_detector import build_detector as build_ssnm
from .SSM.build_detector import build_detector as build_ssm
from .TSNM.build_detector import build_detector as build_tsnm
from .TSM.build_detector import build_ssd_detector_with_metadata as build_tsm


def load_config():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "config.json"
    with open(config_path, "r") as f:
        return json.load(f), project_root


def make_grid_encoder(num_classes, image_size, grid_size):
    """Create encoder for grid format."""
    h, w = image_size
    if isinstance(grid_size, (tuple, list)):
        grid_h, grid_w = grid_size
    else:
        grid_h = grid_w = grid_size
    
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


def compute_detection_metrics(predictions_list, ground_truth_list, num_classes, iou_threshold=0.5):
    """
    Compute comprehensive detection metrics including TP, FP, FN, Precision, Recall, and Match Rate.
    
    Uses IoU matching to determine true positives:
    - TP: Prediction with IoU >= threshold and correct class
    - FP: Prediction with IoU < threshold or wrong class
    - FN: GT box not matched by any prediction
    
    Args:
        predictions_list: List of lists, each containing prediction dicts for one image
        ground_truth_list: List of lists, each containing GT dicts for one image
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching predictions to GT
    
    Returns:
        Dictionary with keys: tp, fp, fn, precision, recall, match_rate, avg_iou_matched
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious = []
    total_predictions = 0
    
    for preds, gts in zip(predictions_list, ground_truth_list):
        # Track which GT boxes have been matched
        matched_gt_indices = set()
        total_predictions += len(preds)
        
        # For each prediction, find best matching GT box
        for pred in preds:
            pred_class_id = pred.get("class_id")
            if pred_class_id < 0 or pred_class_id >= num_classes:
                total_fp += 1
                continue
            
            best_iou = 0.0
            best_gt_idx = None
            
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt_indices:
                    continue
                
                # Check class match
                if pred_class_id == gt.get("class_id"):
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx is not None:
                # True positive: matched GT box with sufficient IoU
                total_tp += 1
                matched_ious.append(best_iou)
                matched_gt_indices.add(best_gt_idx)
            else:
                # False positive: no match or insufficient IoU
                total_fp += 1
        
        # False negatives: unmatched GT boxes
        total_fn += len(gts) - len(matched_gt_indices)
    
    # Compute precision and recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    # Match rate: what fraction of predictions are correct matches
    match_rate = total_tp / total_predictions if total_predictions > 0 else 0.0
    
    # Average IoU of matched predictions
    avg_iou_matched = np.mean(matched_ious) if len(matched_ious) > 0 else 0.0
    
    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "match_rate": match_rate,
        "avg_iou_matched": avg_iou_matched,
    }


def compute_class_only_metrics(predictions_list, ground_truth_list, num_classes):
    """
    Compute detection metrics based on class correctness only (no IoU requirement).
    
    This shows if models are at least predicting the correct classes, regardless of
    bounding box accuracy.
    
    Uses class matching to determine true positives:
    - TP: Prediction with correct class (matched to any GT box of same class)
    - FP: Prediction with wrong class or no matching GT class
    - FN: GT box not matched by any prediction of the same class
    
    Args:
        predictions_list: List of lists, each containing prediction dicts for one image
        ground_truth_list: List of lists, each containing GT dicts for one image
        num_classes: Number of classes
    
    Returns:
        Dictionary with keys: tp, fp, fn, precision, recall, match_rate
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predictions = 0
    
    for preds, gts in zip(predictions_list, ground_truth_list):
        # Track which GT boxes have been matched (by class)
        matched_gt_indices = set()
        total_predictions += len(preds)
        
        # For each prediction, find any matching GT box of the same class
        for pred in preds:
            pred_class_id = pred.get("class_id")
            if pred_class_id < 0 or pred_class_id >= num_classes:
                total_fp += 1
                continue
            
            # Find any unmatched GT box with the same class
            matched = False
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gt_indices:
                    continue
                
                # Check class match only (no IoU requirement)
                if pred_class_id == gt.get("class_id"):
                    # True positive: correct class
                    total_tp += 1
                    matched_gt_indices.add(gt_idx)
                    matched = True
                    break
            
            if not matched:
                # False positive: no GT box of this class available
                total_fp += 1
        
        # False negatives: unmatched GT boxes (no prediction of the same class)
        total_fn += len(gts) - len(matched_gt_indices)
    
    # Compute precision and recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    # Match rate: what fraction of predictions have correct class
    match_rate = total_tp / total_predictions if total_predictions > 0 else 0.0
    
    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "match_rate": match_rate,
    }


def load_model(model_name, model_path, config, num_classes=None, image_size=None):
    """
    Load a saved model with appropriate custom objects.
    
    Args:
        model_name: Name of the model (SSNM, SSM, TSNM, TSM)
        model_path: Path to the model file
        config: Configuration dictionary
        num_classes: Number of classes (required for building model if loading fails)
        image_size: Image size tuple (required for building model if loading fails)
    """
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 4.0)
    bbox_loss_weight = config.get("bbox_loss_weight", 10.0)
    objectness_label_smoothing = config.get("objectness_label_smoothing", 0.1)
    
    # Get anchor configuration for loss function
    num_anchors = config.get("num_anchors", 3)
    use_anchors = config.get("use_anchors", True)
    
    loss_fn = DetectionLossFocal(
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        positive_weight=positive_weight,
        bbox_loss_weight=bbox_loss_weight,
        num_anchors=num_anchors,
        use_anchors=use_anchors,
        objectness_label_smoothing=objectness_label_smoothing,
    )
    component_metrics = make_component_loss_metrics(loss_fn)
    
    
    custom_objects = {
        "DetectionLossFocal": DetectionLossFocal,
        "objectness_accuracy": objectness_accuracy,
        "FiLMLayer": FiLMLayer,
    }
    for metric_fn in component_metrics:
        custom_objects[metric_fn.__name__] = metric_fn
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"  ✓ Loaded {model_name} from Keras format")
        return model
    except Exception as e:
        print(f"  ✗ Could not load {model_name} from Keras format: {e}")
        print(f"  Trying to load weights only...")
        
        # Need num_classes and image_size to build model
        if num_classes is None or image_size is None:
            print(f"  Error: num_classes and image_size required to build model for weights loading")
            return None
        
        # Build model with correct architecture
        pretrained_model_type = config.get("pretrained_model_type", "efficientnet_b0")
        metadata_dim = 8  # Standard metadata dimension
        
        try:
            if model_name == "SSNM":
                model = build_ssnm(
                    image_size=image_size,
                    num_classes=num_classes,
                    num_anchors=num_anchors,
                    backbone_type=pretrained_model_type,
                )
            elif model_name == "SSM":
                model = build_ssm(
                    image_size=image_size,
                    num_classes=num_classes,
                    metadata_dim=metadata_dim,
                    backbone_type=pretrained_model_type,
                    num_anchors=num_anchors,
                )
            elif model_name == "TSNM":
                model = build_tsnm(
                    image_size=image_size,
                    num_classes=num_classes,
                    backbone_type=pretrained_model_type,
                    freeze_backbone=False,  # For evaluation
                    num_anchors=num_anchors,
                )
            elif model_name == "TSM":
                model = build_tsm(
                    image_size=image_size,
                    num_classes=num_classes,
                    metadata_dim=metadata_dim,
                    backbone_type=pretrained_model_type,
                    freeze_backbone=False,  # For evaluation
                    num_anchors=num_anchors,
                )
            else:
                print(f"  Error: Unknown model name: {model_name}")
                return None
            
            # Check for weights file with the pattern used by SafeModelCheckpoint
            weights_path = model_path.parent / f"{model_name}_best_weights.h5"
            if not weights_path.exists():
                weights_path = model_path.with_suffix('.h5')
            
            if weights_path.exists():
                model.load_weights(str(weights_path))
                print(f"  ✓ Loaded weights from {weights_path}")
                # Ensure model is in inference mode
                # Build the model with a dummy input to ensure all layers are properly initialized
                dummy_input = tf.zeros((1, image_size[0], image_size[1], 3))
                # Check if model needs metadata based on number of inputs
                if len(model.inputs) == 2:
                    dummy_meta = tf.zeros((1, 8))
                    _ = model([dummy_input, dummy_meta], training=False)
                else:
                    _ = model(dummy_input, training=False)
                return model
            else:
                print(f"  ✗ Weights file not found:")
                print(f"    Looked for: {model_path.parent / f'{model_name}_best_weights.h5'}")
                print(f"    Looked for (fallback): {model_path.with_suffix('.h5')}")
                return None
        except Exception as build_e:
            print(f"  ✗ Error building model: {build_e}")
            return None


def evaluate_model(model, model_name, val_ds, num_classes, class_names, grid_size, 
                   num_anchors=1, use_anchors=False, max_batches=20):
    """Evaluate a single model and return metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Check if model needs metadata
    needs_metadata = len(model.inputs) == 2
    
    predictions_list = []
    ground_truth_list = []
    
    batch_count = 0
    total_gt_found = 0
    for batch in val_ds:
        if batch_count >= max_batches:
            break
        
        if isinstance(batch, tuple) and len(batch) == 2:
            inputs, targets = batch
            
            if needs_metadata:
                # Model expects [images, metadata]
                if isinstance(inputs, tuple):
                    images, metadata = inputs
                else:
                    quit(400)
                    # This shouldn't happen if dataset is properly prepared, but handle it
                    # Extract metadata from targets if available
                    from .utils.metadata_encoding import encode_metadata_from_tfrecords
                    batch_size = tf.shape(inputs)[0]
                    
                    # Try to get metadata from targets
                    location_str = targets.get("location", tf.fill([batch_size], ""))
                    date_captured_str = targets.get("date_captured", tf.fill([batch_size], "2000-01-01 12:00:00"))
                    
                    # Ensure proper shape
                    location_str = tf.reshape(location_str, [batch_size])
                    date_captured_str = tf.reshape(date_captured_str, [batch_size])
                    
                    # Handle empty strings
                    location_str = tf.where(
                        tf.equal(tf.strings.length(location_str), 0),
                        tf.fill([batch_size], "0"),
                        location_str
                    )
                    date_captured_str = tf.where(
                        tf.equal(tf.strings.length(date_captured_str), 0),
                        tf.fill([batch_size], "2000-01-01 12:00:00"),
                        date_captured_str
                    )
                    
                    # Encode metadata
                    metadata = encode_metadata_from_tfrecords(inputs, location_str, date_captured_str)
                    images = inputs
            else:
                # Model expects just images
                images = inputs
                metadata = None
        else:
            continue
        
        # Preprocess images
        if images.dtype == tf.float32 and tf.reduce_max(images) <= 1.0:
            images_for_model = images * 255.0
        else:
            images_for_model = images
        
        # Get predictions
        if needs_metadata:
            pred_grids = model([images_for_model, metadata], training=False)
        else:
            pred_grids = model(images_for_model, training=False)
        
        pred_grids_np = pred_grids.numpy()
        
        # Decode predictions
        batch_size_actual = pred_grids_np.shape[0]
        total_before_filter = 0
        total_after_filter = 0
        
        # Diagnostic: inspect prediction values for first image of first batch
        if batch_count == 0 and len(predictions_list) == 0:
            first_pred_grid = pred_grids_np[0]
            if use_anchors and num_anchors > 1:
                # For anchor-based: [H, W, num_anchors * (1 + 4 + num_classes)]
                H, W = grid_size
                depth_per_anchor = 1 + 4 + num_classes
                obj_values = []
                class_max_values = []
                for y in range(H):
                    for x in range(W):
                        cell = first_pred_grid[y, x]
                        for a in range(num_anchors):
                            anchor_start = a * depth_per_anchor
                            obj = cell[anchor_start]
                            class_probs = cell[anchor_start + 5:anchor_start + 5 + num_classes]
                            class_max = np.max(class_probs)
                            obj_values.append(obj)
                            class_max_values.append(class_max)
                print(f"  [Debug] First image prediction stats:")
                print(f"    Objectness: min={np.min(obj_values):.3f}, max={np.max(obj_values):.3f}, mean={np.mean(obj_values):.3f}, median={np.median(obj_values):.3f}")
                print(f"    Class prob (max): min={np.min(class_max_values):.3f}, max={np.max(class_max_values):.3f}, mean={np.mean(class_max_values):.3f}, median={np.median(class_max_values):.3f}")
                print(f"    Values >= 0.7: obj={np.sum(np.array(obj_values) >= 0.7)}, class={np.sum(np.array(class_max_values) >= 0.7)}")
            else:
                # For grid-based: [H, W, 1 + 4 + num_classes]
                obj_values = first_pred_grid[:, :, 0].flatten()
                class_probs = first_pred_grid[:, :, 5:].reshape(-1, num_classes)
                class_max_values = np.max(class_probs, axis=1)
                print(f"  [Debug] First image prediction stats:")
                print(f"    Objectness: min={np.min(obj_values):.3f}, max={np.max(obj_values):.3f}, mean={np.mean(obj_values):.3f}, median={np.median(obj_values):.3f}")
                print(f"    Class prob (max): min={np.min(class_max_values):.3f}, max={np.max(class_max_values):.3f}, mean={np.mean(class_max_values):.3f}, median={np.median(class_max_values):.3f}")
                print(f"    Values >= 0.7: obj={np.sum(obj_values >= 0.7)}, class={np.sum(class_max_values >= 0.7)}")
        
        for i in range(batch_size_actual):
            pred_grid = pred_grids_np[i]
            
            if use_anchors and num_anchors > 1:
                pred_boxes = decode_predictions_anchors(
                    pred_grid,
                    num_classes=num_classes,
                    grid_size=grid_size,
                    num_anchors=num_anchors,
                    threshold=0.01,  # Lower threshold to catch more predictions
                    nms_iou=0.5,
                    max_boxes=5,  # Allow more predictions per image
                )
            else:
                pred_boxes = decode_predictions_grid(
                    pred_grid,
                    num_classes=num_classes,
                    threshold=0.01,  # Lower threshold to catch more predictions
                    nms_iou=0.5,
                    max_boxes=5,  # Allow more predictions per image
                )
            
            total_before_filter += len(pred_boxes)
            
            # Filter by combined score (objectness * class_prob) to reduce false positives
            # Lower threshold to allow more predictions through for evaluation
            score_threshold = 0.01  # Lower combined score threshold (objectness * class_prob)
            pred_boxes = [
                box for box in pred_boxes 
                if box.get("score", 0.0) >= score_threshold
            ]
            
            total_after_filter += len(pred_boxes)
            predictions_list.append(pred_boxes)
            
            # Get ground truth from raw dataset format (dict with bboxes and labels)
            gt_boxes = []
            if isinstance(targets, dict):
                # Extract bboxes and labels for this image
                bboxes_batch = targets["bboxes"]
                labels_batch = targets["labels"]
                
                # Convert to numpy if needed
                if hasattr(bboxes_batch, "numpy"):
                    bboxes = bboxes_batch.numpy()[i]
                    labels = labels_batch.numpy()[i]
                else:
                    bboxes = bboxes_batch[i]
                    labels = labels_batch[i]
                
                # Filter out padded boxes (zero boxes)
                for bbox, label in zip(bboxes, labels):
                    # Check if bbox is valid (not all zeros)
                    # Also check if box has valid dimensions
                    ymin, xmin, ymax, xmax = bbox
                    if (np.sum(np.abs(bbox)) > 1e-6 and  # Not all zeros
                        ymax > ymin and xmax > xmin and  # Valid dimensions
                        ymin >= 0 and xmin >= 0 and ymax <= 1 and xmax <= 1):  # Within bounds
                        gt_boxes.append({
                            "bbox": [float(ymin), float(xmin), float(ymax), float(xmax)],
                            "class_id": int(label),
                        })
            else:
                # If targets are not in dict format, try to extract from tensor/array
                # This shouldn't happen with raw dataset, but handle it just in case
                print(f"  [Warning] Unexpected targets format: {type(targets)}")
            
            ground_truth_list.append(gt_boxes)
            total_gt_found += len(gt_boxes)
        
        batch_count += 1
        
        # Debug output for first batch
        if batch_count == 1:
            print(f"  [Debug] After first batch: {total_before_filter} boxes before score filter, {total_after_filter} after")
        
        # Print summary after all batches
        if batch_count >= max_batches:
            print(f"  [Debug] Total: {total_before_filter} boxes before score filter, {total_after_filter} after (across {batch_count} batches)")
        
        if batch_count == 1:
            print(f"  [Debug] First batch:")
            print(f"    Targets type: {type(targets)}")
            if isinstance(targets, dict):
                print(f"    Targets keys: {targets.keys()}")
                if "bboxes" in targets:
                    bboxes_shape = targets["bboxes"].shape if hasattr(targets["bboxes"], "shape") else "unknown"
                    print(f"    Bboxes shape: {bboxes_shape}")
                    if hasattr(targets["bboxes"], "numpy"):
                        bboxes_np = targets["bboxes"].numpy()
                        print(f"    Bboxes numpy shape: {bboxes_np.shape}")
                        print(f"    Non-zero bboxes in first image: {np.sum(np.any(bboxes_np[0] != 0, axis=1))}")
                if "labels" in targets:
                    labels_shape = targets["labels"].shape if hasattr(targets["labels"], "shape") else "unknown"
                    print(f"    Labels shape: {labels_shape}")
            print(f"    GT boxes found in first batch: {sum(len(gt) for gt in ground_truth_list)}")
    
    # Compute metrics with IoU requirement
    iou_threshold = 0.5
    map_score = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=iou_threshold)
    detection_metrics = compute_detection_metrics(predictions_list, ground_truth_list, num_classes, iou_threshold=iou_threshold)
    
    # Compute class-only metrics (no IoU requirement)
    class_only_metrics = compute_class_only_metrics(predictions_list, ground_truth_list, num_classes)
    
    total_gt = sum(len(gt) for gt in ground_truth_list)
    total_pred = sum(len(pred) for pred in predictions_list)
    
    results = {
        "model": model_name,
        "mAP": map_score,
        "precision": detection_metrics["precision"],
        "recall": detection_metrics["recall"],
        "tp": detection_metrics["tp"],
        "fp": detection_metrics["fp"],
        "fn": detection_metrics["fn"],
        "match_rate": detection_metrics["match_rate"],
        "avg_iou_matched": detection_metrics["avg_iou_matched"],
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "avg_pred_per_image": total_pred / len(predictions_list) if predictions_list else 0,
        "avg_gt_per_image": total_gt / len(ground_truth_list) if ground_truth_list else 0,
        # Class-only metrics
        "class_only_precision": class_only_metrics["precision"],
        "class_only_recall": class_only_metrics["recall"],
        "class_only_tp": class_only_metrics["tp"],
        "class_only_fp": class_only_metrics["fp"],
        "class_only_fn": class_only_metrics["fn"],
        "class_only_match_rate": class_only_metrics["match_rate"],
    }
    
    print(f"  mAP@{iou_threshold}: {map_score:.4f}")
    print(f"  Precision: {detection_metrics['precision']:.4f}")
    print(f"  Recall: {detection_metrics['recall']:.4f}")
    print(f"  TP: {detection_metrics['tp']}, FP: {detection_metrics['fp']}, FN: {detection_metrics['fn']}")
    print(f"  Match Rate: {detection_metrics['match_rate']:.4f} ({detection_metrics['tp']}/{total_pred} predictions matched)")
    print(f"  Avg IoU (matched): {detection_metrics['avg_iou_matched']:.4f}")
    print(f"  Total GT boxes: {total_gt}")
    print(f"  Total pred boxes: {total_pred}")
    print(f"  Images evaluated: {len(predictions_list)}")
    
    if total_gt == 0:
        print(f"  ⚠ WARNING: No ground truth boxes found! Check dataset encoding.")
    
    return results


def main():
    config, project_root = load_config()
    
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    models_dir = project_root / config["models_dir"]
    
    print("=" * 80)
    print("Model Comparison: SSNM vs SSM vs TSNM vs TSM")
    print("=" * 80)
    
    # Load dataset
    dataset_name = config.get("dataset", "cct")
    if dataset_name != "cct":
        raise ValueError(f"Comparison currently only supports CCT dataset")
    
    cct_tfrecords_dir = config.get("cct_tfrecords_dir")
    if not cct_tfrecords_dir or not Path(cct_tfrecords_dir).exists():
        raise ValueError(f"TFRecords directory not found: {cct_tfrecords_dir}")
    
    print(f"\n[Dataset] Using TFRecords from {cct_tfrecords_dir}")
    val_ds_raw, val_info = make_cct_tfrecords_dataset(
        tfrecords_dir=cct_tfrecords_dir,
        split="val",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    
    num_classes = val_info.features["objects"]["label"].num_classes
    class_names = val_info.features["objects"]["label"].names
    print(f"Number of classes: {num_classes}")
    
    # Prepare datasets for each model type
    # Use raw dataset format (not encoded) for proper GT extraction
    # Models will handle their own preprocessing
    
    # For models without metadata: use raw dataset
    val_ds_no_meta = val_ds_raw.prefetch(tf.data.AUTOTUNE)
    
    # For models with metadata: extract and encode actual metadata from TFRecords
    from .utils.metadata_encoding import encode_metadata_from_tfrecords
    
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
        
        # Keep targets as-is (with metadata strings) for GT extraction
        return (images, metadata), targets
    
    val_ds_with_meta = val_ds_raw.map(add_metadata, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # Get grid size and anchor configuration
    num_anchors = config.get("num_anchors", 3)
    use_anchors = config.get("use_anchors", True)  # Default to True for new models
    
    # Infer grid size from image size (CSPDarkNet downsamples by 32x)
    grid_h = image_size[0] // 32
    grid_w = image_size[1] // 32
    grid_size = (grid_h, grid_w)
    print(f"Grid size: {grid_h}x{grid_w}, num_anchors: {num_anchors}, use_anchors: {use_anchors}")
    
    # Models to evaluate
    models_to_eval = [
        ("SSNM", models_dir / "SSNM_best.keras", False, val_ds_no_meta, use_anchors),
        ("SSM", models_dir / "SSM_best.keras", True, val_ds_with_meta, use_anchors),
        ("TSNM", models_dir / "TSNM_best.keras", False, val_ds_no_meta, use_anchors),
        ("TSM", models_dir / "TSM_best.keras", True, val_ds_with_meta, use_anchors),
    ]
    
    all_results = []
    
    for model_name, model_path, needs_metadata, val_ds, use_anchors_flag in models_to_eval:
        if not model_path.exists():
            print(f"\n{model_name}: Model file not found at {model_path}")
            print("  Skipping evaluation")
            continue
        
        # Load model (pass num_classes and image_size for fallback weight loading)
        model = load_model(model_name, model_path, config, num_classes=num_classes, image_size=image_size)
        if model is None:
            continue
        
        # Evaluate
        results = evaluate_model(model, model_name, val_ds, num_classes, class_names, grid_size,
                                num_anchors=num_anchors, use_anchors=use_anchors_flag)
        all_results.append(results)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    if all_results:
        if HAS_PANDAS:
            df = pd.DataFrame(all_results)
            print("\n" + df.to_string(index=False))
            
            # Print class-only comparison table
            print("\n" + "=" * 80)
            print("CLASS-ONLY COMPARISON SUMMARY (No IoU Requirement)")
            print("=" * 80)
            print("This table shows metrics based on class correctness only, regardless of bounding box accuracy.")
            class_only_cols = ['model', 'class_only_precision', 'class_only_recall', 'class_only_tp', 
                              'class_only_fp', 'class_only_fn', 'class_only_match_rate']
            class_only_df = df[class_only_cols].copy()
            class_only_df.columns = ['Model', 'Precision', 'Recall', 'True Positives', 
                                     'False Positives', 'False Negatives', 'Match Rate']
            print("\n" + class_only_df.to_string(index=False))
            
            # Find best model for each metric
            print("\n" + "=" * 80)
            print("BEST MODELS BY METRIC")
            print("=" * 80)
            print(f"  Best mAP: {df.loc[df['mAP'].idxmax(), 'model']} ({df['mAP'].max():.4f})")
            print(f"  Best Precision: {df.loc[df['precision'].idxmax(), 'model']} ({df['precision'].max():.4f})")
            print(f"  Best Recall: {df.loc[df['recall'].idxmax(), 'model']} ({df['recall'].max():.4f})")
            print(f"  Best Match Rate: {df.loc[df['match_rate'].idxmax(), 'model']} ({df['match_rate'].max():.4f})")
            print(f"\n  Best Class-Only Precision: {df.loc[df['class_only_precision'].idxmax(), 'model']} ({df['class_only_precision'].max():.4f})")
            print(f"  Best Class-Only Recall: {df.loc[df['class_only_recall'].idxmax(), 'model']} ({df['class_only_recall'].max():.4f})")
            print(f"  Best Class-Only Match Rate: {df.loc[df['class_only_match_rate'].idxmax(), 'model']} ({df['class_only_match_rate'].max():.4f})")
            
            # Save results
            results_path = project_root / "model_comparison_results.csv"
            df.to_csv(results_path, index=False)
            print(f"\nResults saved to: {results_path}")
        else:
            # Print without pandas
            print("\nModel Comparison Results:")
            print("-" * 120)
            print(f"{'Model':<8} {'Mean AP':<10} {'Precision':<12} {'Recall':<10} {'True Positives':<16} {'False Positives':<16} {'False Negatives':<16} {'Match Rate':<12} {'Total Predictions':<18}")
            print("-" * 120)
            for r in all_results:
                match_pct = r['match_rate'] * 100
                print(f"{r['model']:<8} {r['mAP']:<10.4f} {r['precision']:<12.4f} {r['recall']:<10.4f} "
                      f"{r['tp']:<16} {r['fp']:<16} {r['fn']:<16} {match_pct:<12.2f} {r['total_pred_boxes']:<18}")
            
            # Print class-only comparison table (no IoU requirement)
            print("\n" + "=" * 80)
            print("CLASS-ONLY COMPARISON SUMMARY (No IoU Requirement)")
            print("=" * 80)
            print("This table shows metrics based on class correctness only, regardless of bounding box accuracy.")
            print("\nClass-Only Comparison Results:")
            print("-" * 120)
            print(f"{'Model':<8} {'Precision':<12} {'Recall':<10} {'True Positives':<16} {'False Positives':<16} {'False Negatives':<16} {'Match Rate':<12}")
            print("-" * 120)
            for r in all_results:
                class_match_pct = r['class_only_match_rate'] * 100
                print(f"{r['model']:<8} {r['class_only_precision']:<12.4f} {r['class_only_recall']:<10.4f} "
                      f"{r['class_only_tp']:<16} {r['class_only_fp']:<16} {r['class_only_fn']:<16} {class_match_pct:<12.2f}")
            
            # Find best models
            print("\n" + "=" * 80)
            print("BEST MODELS BY METRIC")
            print("=" * 80)
            best_map = max(all_results, key=lambda x: x['mAP'])
            best_prec = max(all_results, key=lambda x: x['precision'])
            best_recall = max(all_results, key=lambda x: x['recall'])
            best_match = max(all_results, key=lambda x: x['match_rate'])
            print(f"  Best mAP: {best_map['model']} ({best_map['mAP']:.4f})")
            print(f"  Best Precision: {best_prec['model']} ({best_prec['precision']:.4f})")
            print(f"  Best Recall: {best_recall['model']} ({best_recall['recall']:.4f})")
            print(f"  Best Match Rate: {best_match['model']} ({best_match['match_rate']:.4f})")
            
            # Save as JSON
            results_path = project_root / "model_comparison_results.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {results_path}")
    else:
        print("\nNo models were successfully evaluated.")


if __name__ == "__main__":
    main()

