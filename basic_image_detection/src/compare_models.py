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
    compute_iou,
)

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


def compute_mean_precision(predictions_list, ground_truth_list, num_classes):
    """
    Compute mean precision across all classes. This metric penalizes incorrect class predictions.
    
    Precision = TP / (TP + FP) for each class, where:
    - TP (True Positive): Prediction with correct class_id that matches a GT box of that class
    - FP (False Positive): Prediction with incorrect class_id (doesn't match any GT box of that class)
    
    Returns the average precision across all classes.
    
    Args:
        predictions_list: List of lists, each containing prediction dicts for one image
        ground_truth_list: List of lists, each containing GT dicts for one image
        num_classes: Number of classes
    
    Returns:
        Mean precision (float): average precision across all classes (0-1)
    """
    # Per-class counts: [TP, FP] for each class
    class_stats = {class_id: {"tp": 0, "fp": 0} for class_id in range(num_classes)}
    
    for preds, gts in zip(predictions_list, ground_truth_list):
        # Get GT class IDs for this image
        gt_class_ids = set(gt.get("class_id") for gt in gts)
        
        # For each prediction, check if it's a TP or FP
        for pred in preds:
            pred_class_id = pred.get("class_id")
            if pred_class_id < 0 or pred_class_id >= num_classes:
                continue
            
            # Check if this class appears in GT for this image
            if pred_class_id in gt_class_ids:
                # True positive: prediction matches a GT class in this image
                class_stats[pred_class_id]["tp"] += 1
            else:
                # False positive: prediction doesn't match any GT class in this image
                class_stats[pred_class_id]["fp"] += 1
    
    # Compute precision for each class
    precisions = []
    for class_id in range(num_classes):
        tp = class_stats[class_id]["tp"]
        fp = class_stats[class_id]["fp"]
        
        if tp + fp == 0:
            # No predictions for this class, skip it
            continue
        
        precision = tp / (tp + fp)
        precisions.append(precision)
    
    # Return mean precision across classes that had predictions
    if len(precisions) == 0:
        return 0.0
    
    return sum(precisions) / len(precisions)


def compute_empty_accuracy(predictions_list, ground_truth_list):
    """
    Compute accuracy of detecting empty images (images with no objects).
    
    Args:
        predictions_list: List of lists, each containing prediction dicts for one image
        ground_truth_list: List of lists, each containing GT dicts for one image
    
    Returns:
        Empty accuracy (float): fraction of empty images that are correctly identified as empty
    """
    total_empty_images = 0
    correctly_identified_empty = 0
    
    for preds, gts in zip(predictions_list, ground_truth_list):
        # GT is empty if there are no GT boxes
        gt_is_empty = len(gts) == 0
        
        if gt_is_empty:
            total_empty_images += 1
            # Prediction is correct if it also predicts no objects
            pred_is_empty = len(preds) == 0
            if pred_is_empty:
                correctly_identified_empty += 1
    
    if total_empty_images == 0:
        return 1.0  # No empty images in dataset, return perfect score
    
    return correctly_identified_empty / total_empty_images


def load_model(model_name, model_path, config):
    """Load a saved model with appropriate custom objects."""
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 4.0)
    bbox_loss_weight = config.get("bbox_loss_weight", 10.0)
    
    loss_fn = DetectionLossFocal(
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        positive_weight=positive_weight,
        bbox_loss_weight=bbox_loss_weight
    )
    component_metrics = make_component_loss_metrics(loss_fn)
    
    custom_objects = {
        "DetectionLossFocal": DetectionLossFocal,
        "objectness_accuracy": objectness_accuracy,
    }
    for metric_fn in component_metrics:
        custom_objects[metric_fn.__name__] = metric_fn
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        print(f"  Warning: Could not load {model_name}: {e}")
        return None


def evaluate_model(model, model_name, val_ds, num_classes, class_names, grid_size, max_batches=20):
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
                    # Need to add metadata
                    batch_size = tf.shape(inputs)[0]
                    brightness = tf.reduce_mean(inputs, axis=[1, 2, 3])
                    location_id = tf.zeros([batch_size], dtype=tf.float32)
                    hour = tf.ones([batch_size], dtype=tf.float32) * 0.5
                    day_of_week = tf.ones([batch_size], dtype=tf.float32) * 0.5
                    month = tf.ones([batch_size], dtype=tf.float32) * 0.5
                    metadata = tf.stack([location_id, hour, day_of_week, month, brightness], axis=1)
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
        for i in range(batch_size_actual):
            pred_grid = pred_grids_np[i]
            
            pred_boxes = decode_predictions_grid(
                pred_grid,
                num_classes=num_classes,
                threshold=0.1,
                nms_iou=0.5,
                max_boxes=50,
            )
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
    
    # Compute metrics
    mean_precision = compute_mean_precision(predictions_list, ground_truth_list, num_classes)
    empty_accuracy = compute_empty_accuracy(predictions_list, ground_truth_list)
    
    total_gt = sum(len(gt) for gt in ground_truth_list)
    total_pred = sum(len(pred) for pred in predictions_list)
    
    results = {
        "model": model_name,
        "mean_precision": mean_precision,
        "empty_accuracy": empty_accuracy,
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "avg_pred_per_image": total_pred / len(predictions_list) if predictions_list else 0,
        "avg_gt_per_image": total_gt / len(ground_truth_list) if ground_truth_list else 0,
    }
    
    print(f"  Mean Precision: {mean_precision:.4f}")
    print(f"  Empty Accuracy: {empty_accuracy:.4f}")
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
    
    # For models with metadata: add metadata but keep raw targets
    def add_placeholder_metadata(images, targets):
        batch_size = tf.shape(images)[0]
        brightness = tf.reduce_mean(images, axis=[1, 2, 3])
        location_id = tf.zeros([batch_size], dtype=tf.float32)
        hour = tf.ones([batch_size], dtype=tf.float32) * 0.5
        day_of_week = tf.ones([batch_size], dtype=tf.float32) * 0.5
        month = tf.ones([batch_size], dtype=tf.float32) * 0.5
        metadata = tf.stack([location_id, hour, day_of_week, month, brightness], axis=1)
        return (images, metadata), targets
    
    val_ds_with_meta = val_ds_raw.map(add_placeholder_metadata, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # Get grid size for decoding predictions (need to know output shape)
    pretrained_model_type = config.get("pretrained_model_type", "ssd_mobilenet_v2")
    test_model = build_ssnm(image_size, num_classes)
    test_output = test_model(tf.zeros((1, image_size[0], image_size[1], 3)), training=False)
    grid_h, grid_w = test_output.shape[1], test_output.shape[2]
    grid_size = (grid_h, grid_w)
    print(f"Grid size: {grid_h}x{grid_w}")
    
    # Models to evaluate
    models_to_eval = [
        ("SSNM", models_dir / "SSNM_best.keras", False, val_ds_no_meta),
        ("SSM", models_dir / "SSM_best.keras", True, val_ds_with_meta),
        ("TSNM", models_dir / "TSNM_best.keras", False, val_ds_no_meta),
        ("TSM", models_dir / "TSM_best.keras", True, val_ds_with_meta),
    ]
    
    all_results = []
    
    for model_name, model_path, needs_metadata, val_ds in models_to_eval:
        if not model_path.exists():
            print(f"\n{model_name}: Model file not found at {model_path}")
            print("  Skipping evaluation")
            continue
        
        # Load model
        model = load_model(model_name, model_path, config)
        if model is None:
            continue
        
        # Evaluate
        results = evaluate_model(model, model_name, val_ds, num_classes, class_names, grid_size)
        all_results.append(results)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    if all_results:
        if HAS_PANDAS:
            df = pd.DataFrame(all_results)
            print("\n" + df.to_string(index=False))
            
            # Find best model for each metric
            print("\n" + "=" * 80)
            print("BEST MODELS BY METRIC")
            print("=" * 80)
            print(f"  Best Mean Precision: {df.loc[df['mean_precision'].idxmax(), 'model']} ({df['mean_precision'].max():.4f})")
            print(f"  Best Empty Accuracy: {df.loc[df['empty_accuracy'].idxmax(), 'model']} ({df['empty_accuracy'].max():.4f})")
            
            # Save results
            results_path = project_root / "model_comparison_results.csv"
            df.to_csv(results_path, index=False)
            print(f"\nResults saved to: {results_path}")
        else:
            # Print without pandas
            print("\nModel Comparison Results:")
            print("-" * 80)
            print(f"{'Model':<8} {'Mean Prec':<12} {'Empty Acc':<12} {'GT Boxes':<10} {'Pred Boxes':<12}")
            print("-" * 80)
            for r in all_results:
                print(f"{r['model']:<8} {r['mean_precision']:<12.4f} {r['empty_accuracy']:<12.4f} {r['total_gt_boxes']:<10} {r['total_pred_boxes']:<12}")
            
            # Find best models
            print("\n" + "=" * 80)
            print("BEST MODELS BY METRIC")
            print("=" * 80)
            best_precision = max(all_results, key=lambda x: x['mean_precision'])
            best_empty = max(all_results, key=lambda x: x['empty_accuracy'])
            print(f"  Best Mean Precision: {best_precision['model']} ({best_precision['mean_precision']:.4f})")
            print(f"  Best Empty Accuracy: {best_empty['model']} ({best_empty['empty_accuracy']:.4f})")
            
            # Save as JSON
            results_path = project_root / "model_comparison_results.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {results_path}")
    else:
        print("\nNo models were successfully evaluated.")


if __name__ == "__main__":
    main()

