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
    compute_map,
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
            
            # Get ground truth from grid targets
            gt_boxes = []
            if isinstance(targets, np.ndarray):
                # Grid format
                grid_t = targets[i]
                S = grid_t.shape[0]
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
                        
                        xmin = np.clip(xmin, 0, 1)
                        ymin = np.clip(ymin, 0, 1)
                        xmax = np.clip(xmax, 0, 1)
                        ymax = np.clip(ymax, 0, 1)
                        
                        class_probs = cell[5:]
                        class_id = int(np.argmax(class_probs))
                        
                        gt_boxes.append({
                            "bbox": [ymin, xmin, ymax, xmax],
                            "class_id": class_id,
                        })
            elif isinstance(targets, dict):
                # Dict format
                bboxes_batch = targets["bboxes"]
                labels_batch = targets["labels"]
                
                if hasattr(bboxes_batch, "numpy"):
                    bboxes = bboxes_batch.numpy()[i]
                    labels = labels_batch.numpy()[i]
                else:
                    bboxes = bboxes_batch[i]
                    labels = labels_batch[i]
                
                for bbox, label in zip(bboxes, labels):
                    ymin, xmin, ymax, xmax = bbox
                    if (np.sum(np.abs(bbox)) > 1e-6 and
                        ymax > ymin and xmax > xmin and
                        ymin >= 0 and xmin >= 0 and ymax <= 1 and xmax <= 1):
                        gt_boxes.append({
                            "bbox": [float(ymin), float(xmin), float(ymax), float(xmax)],
                            "class_id": int(label),
                        })
            
            ground_truth_list.append(gt_boxes)
        
        batch_count += 1
    
    # Compute metrics
    map_3 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.3)
    map_5 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.5)
    map_8 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.8)
    
    total_gt = sum(len(gt) for gt in ground_truth_list)
    total_pred = sum(len(pred) for pred in predictions_list)
    
    results = {
        "model": model_name,
        "mAP@0.3": map_3,
        "mAP@0.5": map_5,
        "mAP@0.8": map_8,
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "avg_pred_per_image": total_pred / len(predictions_list) if predictions_list else 0,
        "avg_gt_per_image": total_gt / len(ground_truth_list) if ground_truth_list else 0,
    }
    
    print(f"  mAP@0.3: {map_3:.4f}")
    print(f"  mAP@0.5: {map_5:.4f}")
    print(f"  mAP@0.8: {map_8:.4f}")
    print(f"  Total GT boxes: {total_gt}")
    print(f"  Total pred boxes: {total_pred}")
    
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
    # Need to get grid size first by building a test model
    pretrained_model_type = config.get("pretrained_model_type", "ssd_mobilenet_v2")
    
    # Build a test model to get grid size
    test_model = build_ssnm(image_size, num_classes)
    test_output = test_model(tf.zeros((1, image_size[0], image_size[1], 3)), training=False)
    grid_h, grid_w = test_output.shape[1], test_output.shape[2]
    grid_size = (grid_h, grid_w)
    print(f"Grid size: {grid_h}x{grid_w}")
    
    # Create encoder
    encoder = make_grid_encoder(num_classes, image_size, grid_size)
    
    # For models without metadata: encode to grid
    val_ds_no_meta = val_ds_raw.map(encoder, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
    # For models with metadata: add metadata then encode
    def add_placeholder_metadata(images, targets):
        batch_size = tf.shape(images)[0]
        brightness = tf.reduce_mean(images, axis=[1, 2, 3])
        location_id = tf.zeros([batch_size], dtype=tf.float32)
        hour = tf.ones([batch_size], dtype=tf.float32) * 0.5
        day_of_week = tf.ones([batch_size], dtype=tf.float32) * 0.5
        month = tf.ones([batch_size], dtype=tf.float32) * 0.5
        metadata = tf.stack([location_id, hour, day_of_week, month, brightness], axis=1)
        return (images, metadata), targets
    
    def encode_with_metadata(inputs, targets):
        images, metadata = inputs
        images_encoded, grid_targets = encoder(images, targets)
        return (images_encoded, metadata), grid_targets
    
    val_ds_with_meta = val_ds_raw.map(add_placeholder_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_with_meta = val_ds_with_meta.map(encode_with_metadata, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    
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
            print(f"  Best mAP@0.3: {df.loc[df['mAP@0.3'].idxmax(), 'model']} ({df['mAP@0.3'].max():.4f})")
            print(f"  Best mAP@0.5: {df.loc[df['mAP@0.5'].idxmax(), 'model']} ({df['mAP@0.5'].max():.4f})")
            print(f"  Best mAP@0.8: {df.loc[df['mAP@0.8'].idxmax(), 'model']} ({df['mAP@0.8'].max():.4f})")
            
            # Save results
            results_path = project_root / "model_comparison_results.csv"
            df.to_csv(results_path, index=False)
            print(f"\nResults saved to: {results_path}")
        else:
            # Print without pandas
            print("\nModel Comparison Results:")
            print("-" * 80)
            print(f"{'Model':<8} {'mAP@0.3':<10} {'mAP@0.5':<10} {'mAP@0.8':<10} {'GT Boxes':<10} {'Pred Boxes':<12}")
            print("-" * 80)
            for r in all_results:
                print(f"{r['model']:<8} {r['mAP@0.3']:<10.4f} {r['mAP@0.5']:<10.4f} {r['mAP@0.8']:<10.4f} {r['total_gt_boxes']:<10} {r['total_pred_boxes']:<12}")
            
            # Find best models
            print("\n" + "=" * 80)
            print("BEST MODELS BY METRIC")
            print("=" * 80)
            best_map3 = max(all_results, key=lambda x: x['mAP@0.3'])
            best_map5 = max(all_results, key=lambda x: x['mAP@0.5'])
            best_map8 = max(all_results, key=lambda x: x['mAP@0.8'])
            print(f"  Best mAP@0.3: {best_map3['model']} ({best_map3['mAP@0.3']:.4f})")
            print(f"  Best mAP@0.5: {best_map5['model']} ({best_map5['mAP@0.5']:.4f})")
            print(f"  Best mAP@0.8: {best_map8['model']} ({best_map8['mAP@0.8']:.4f})")
            
            # Save as JSON
            results_path = project_root / "model_comparison_results.json"
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to: {results_path}")
    else:
        print("\nNo models were successfully evaluated.")


if __name__ == "__main__":
    main()

