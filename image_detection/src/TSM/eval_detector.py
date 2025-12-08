"""
Evaluate YOLO-style detection model with metadata integration.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..pipelines.cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from .build_detector import build_ssd_detector_with_metadata
from ..utils.detection_utils import decode_predictions_grid, decode_predictions_anchors, compute_map, compute_iou, DetectionLossFocal, objectness_accuracy, make_component_loss_metrics


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "config.json"
    
    with open(config_path, "r") as f:
        return json.load(f), project_root


def draw_boxes(image, gt_boxes, pred_boxes, class_names):
    """Draw ground truth and predicted boxes on image."""
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    h, w, _ = image.shape
    
    # Ground truth (green)
    for gt_box in gt_boxes:
        if isinstance(gt_box, dict):
            bbox = gt_box["bbox"]
        else:
            bbox = gt_box
        ymin, xmin, ymax, xmax = bbox
        rect = patches.Rectangle(
            (xmin*w, ymin*h),
            (xmax-xmin)*w,
            (ymax-ymin)*h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        if isinstance(gt_box, dict) and "class_id" in gt_box:
            class_name = class_names[gt_box["class_id"]] if gt_box["class_id"] < len(class_names) else f"Class {gt_box['class_id']}"
            ax.text(
                xmin*w,
                ymin*h - 5,
                f"GT: {class_name}",
                color="lime",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.7),
            )
    
    # Predictions (red)
    for p in pred_boxes:
        ymin, xmin, ymax, xmax = p["bbox"]
        rect = patches.Rectangle(
            (xmin*w, ymin*h),
            (xmax-xmin)*w,
            (ymax-ymin)*h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        class_name = class_names[p["class_id"]] if p["class_id"] < len(class_names) else f"Class {p['class_id']}"
        ax.text(
            xmin*w,
            ymin*h,
            f"{class_name} ({p['score']:.2f})",
            color="red",
            fontsize=8,
            bbox=dict(facecolor="yellow", alpha=0.7),
        )
    
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    config, project_root = load_config()
    
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    
    # Model name is set here, not from config
    model_name = "TSM"
    models_dir = project_root / config["models_dir"]
    model_path = models_dir / f"{model_name}_best.keras"
    
    pretrained_model_type = config.get("pretrained_model_type", "cspdarknet")
    num_anchors = config.get("num_anchors", 3)
    use_anchors = config.get("use_anchors", True)
    
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 4.0)
    bbox_loss_weight = config.get("bbox_loss_weight", 10.0)
    
    print("Loading model:", model_path)
    
    loss_fn = DetectionLossFocal(
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        positive_weight=positive_weight,
        bbox_loss_weight=bbox_loss_weight,
        num_anchors=num_anchors,
        use_anchors=use_anchors,
    )
    
    # Create component loss metrics (needed for loading model that was saved with these metrics)
    component_metrics = make_component_loss_metrics(loss_fn)
    
    # Import FiLMLayer for model loading
    from ..utils.film_layer import FiLMLayer
    
    # Build custom_objects dict with all metrics
    custom_objects = {
        "DetectionLossFocal": DetectionLossFocal,
        "objectness_accuracy": objectness_accuracy,
        "FiLMLayer": FiLMLayer,
    }
    
    # Add component loss metrics
    for metric_fn in component_metrics:
        custom_objects[metric_fn.__name__] = metric_fn
    
    # Load dataset FIRST to get num_classes (needed to build model with correct architecture)
    dataset_name = config.get("dataset", "cct")
    metadata_dim = 8  # Updated: 8 features with cyclical encoding
    
    if dataset_name != "cct":
        raise ValueError(f"YOLO detector currently only supports CCT dataset")
    
    # Load validation dataset from TFRecords
    cct_tfrecords_dir = config.get("cct_tfrecords_dir")
    if not cct_tfrecords_dir or not Path(cct_tfrecords_dir).exists():
        raise ValueError(f"TFRecords directory not found: {cct_tfrecords_dir}. Please generate TFRecords first.")
    
    # Get split name from config
    val_split = config.get("val_split", "val")
    val_split_base = val_split.split("[")[0].strip() if "[" in val_split else val_split
    
    # Map common split name variations to TFRecord naming convention
    split_name_map = {
        "train": "train",
        "training": "train",
        "val": "val",
        "validation": "val",
        "valid": "val",
        "test": "test",
        "testing": "test",
    }
    val_split_base = split_name_map.get(val_split_base.lower(), val_split_base)
    
    print(f"\n[Dataset] Using TFRecords from {cct_tfrecords_dir}")
    print(f"[Dataset] Validation split: {val_split_base}")
    val_ds_raw, val_info = make_cct_tfrecords_dataset(
        tfrecords_dir=cct_tfrecords_dir,
        split=val_split_base,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    
    num_classes = val_info.features["objects"]["label"].num_classes
    class_names = val_info.features["objects"]["label"].names
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Now try to load model (with correct num_classes known)
    print(f"\n[Model] Attempting to load model from {model_path}")
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
        )
        print(f"✓ Model loaded successfully from Keras format!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Trying to load weights only...")
        # Build model with correct num_classes from dataset
        model = build_ssd_detector_with_metadata(
            image_size=image_size,
            num_classes=num_classes,
            metadata_dim=metadata_dim,
            backbone_type=pretrained_model_type,
            freeze_backbone=False,  # For evaluation, backbone can be trainable
            num_anchors=num_anchors,
        )
        # Check for weights file with the pattern used by SafeModelCheckpoint
        weights_path = model_path.parent / f"{model_name}_best_weights.h5"
        if not weights_path.exists():
            weights_path = model_path.with_suffix('.h5')
        
        if weights_path.exists():
            model.load_weights(str(weights_path))
            print(f"✓ Loaded weights from {weights_path}")
        else:
            print(f"✗ Error: Neither model nor weights file found.")
            print(f"  Looked for: {model_path}")
            print(f"  Looked for weights: {model_path.parent / f'{model_name}_best_weights.h5'}")
            print(f"  Looked for weights (fallback): {model_path.with_suffix('.h5')}")
            return
    
    model.summary()
    
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
    
    val_ds_with_meta = val_ds_raw.map(add_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating model...")
    print("=" * 60)
    
    # Collect predictions, ground truth, and images for visualization
    predictions_list = []
    ground_truth_list = []
    images_list = []  # Store images for visualization
    pred_grids_all = []  # Store all prediction grids for diagnostics
    
    for batch_idx, batch in enumerate(val_ds_with_meta):
        if isinstance(batch, tuple) and len(batch) == 2:
            inputs, targets = batch
            if isinstance(inputs, tuple):
                images, metadata = inputs
            else:
                images = inputs
                metadata = None
        else:
            continue
        
        # Ensure images are in correct format for model preprocessing
        # Model expects [0, 255] range for mobilenet_v2.preprocess_input
        # Dataset provides [0, 1] range, so convert if needed
        if images.dtype == tf.float32 and tf.reduce_max(images) <= 1.0:
            # Convert from [0, 1] to [0, 255] for preprocessing
            images_for_model = images * 255.0
        else:
            images_for_model = images
        
        # Get predictions
        if metadata is not None:
            pred_grids = model([images_for_model, metadata], training=False)
        else:
            pred_grids = model(images_for_model, training=False)
        
        # Convert to numpy
        pred_grids_np = pred_grids.numpy()
        images_np = images.numpy()
        
        # Store for diagnostics
        pred_grids_all.append(pred_grids_np)
        
        # Detailed diagnostics for first batch
        if batch_idx == 0:
            print(f"\n[Debug] First batch predictions:")
            print(f"  Prediction shape: {pred_grids_np.shape}")
            print(f"  Prediction dtype: {pred_grids_np.dtype}")
            print(f"  Objectness channel stats:")
            obj_channel = pred_grids_np[..., 0]  # Objectness is first channel
            print(f"    Min: {np.min(obj_channel):.6f}, Max: {np.max(obj_channel):.6f}")
            print(f"    Mean: {np.mean(obj_channel):.6f}, Std: {np.std(obj_channel):.6f}")
            print(f"    Non-zero count: {np.count_nonzero(obj_channel)} / {obj_channel.size}")
            print(f"  Bbox channel stats (cx, cy, w, h):")
            for i, name in enumerate(['cx', 'cy', 'w', 'h']):
                bbox_ch = pred_grids_np[..., 1 + i]
                print(f"    {name}: min={np.min(bbox_ch):.6f}, max={np.max(bbox_ch):.6f}, mean={np.mean(bbox_ch):.6f}")
                # Show values for cells with high objectness
                high_obj_mask = obj_channel > 0.5
                if np.any(high_obj_mask):
                    high_obj_bbox = bbox_ch[high_obj_mask]
                    print(f"      (where obj>0.5): min={np.min(high_obj_bbox):.6f}, max={np.max(high_obj_bbox):.6f}, mean={np.mean(high_obj_bbox):.6f}")
            print(f"  Image stats:")
            print(f"    Min: {np.min(images_np):.6f}, Max: {np.max(images_np):.6f}")
            print(f"    Mean: {np.mean(images_np):.6f}, Std: {np.std(images_np):.6f}")
            if metadata is not None:
                metadata_np = metadata.numpy() if hasattr(metadata, 'numpy') else metadata
                print(f"  Metadata stats:")
                print(f"    Shape: {metadata_np.shape}")
                print(f"    Min: {np.min(metadata_np):.6f}, Max: {np.max(metadata_np):.6f}")
                print(f"    Mean: {np.mean(metadata_np):.6f}")
        
        # Decode predictions for each image in batch
        batch_size_actual = pred_grids_np.shape[0]
        for i in range(batch_size_actual):
            pred_grid = pred_grids_np[i]  # [H, W, 5 + num_classes]
            image_np = images_np[i]  # [H, W, 3]
            
            # Convert image from normalized [0, 1] to [0, 255] uint8 if needed
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            # Decode to boxes
            # Use reasonable threshold - model is now producing good objectness scores
            if use_anchors and num_anchors > 1:
                # Infer grid size from image size
                grid_h = image_size[0] // 32
                grid_w = image_size[1] // 32
                grid_size = (grid_h, grid_w)
                pred_boxes = decode_predictions_anchors(
                    pred_grid,
                    num_classes=num_classes,
                    grid_size=grid_size,
                    num_anchors=num_anchors,
                    threshold=0.1,  # Reasonable threshold to filter low-confidence predictions
                    nms_iou=0.5,  # Standard NMS IoU threshold
                    max_boxes=50,  # Allow more boxes to see if we're missing matches
                )
            else:
                pred_boxes = decode_predictions_grid(
                    pred_grid,
                    num_classes=num_classes,
                    threshold=0.1,  # Reasonable threshold to filter low-confidence predictions
                    nms_iou=0.5,  # Standard NMS IoU threshold
                    max_boxes=50,  # Allow more boxes to see if we're missing matches
                )
            predictions_list.append(pred_boxes)
            
            # Get ground truth boxes
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
                            "bbox": [float(ymin), float(xmin), float(ymax), float(xmax)],  # [ymin, xmin, ymax, xmax]
                            "class_id": int(label),
                        })
            
            ground_truth_list.append(gt_boxes)
            images_list.append(image_np)  # Store image for visualization
        
        # Limit evaluation to first few batches for speed
        if batch_idx >= 10:
            break
    
    # Diagnostic output before computing mAP
    total_gt_boxes = sum(len(gt) for gt in ground_truth_list)
    total_pred_boxes = sum(len(pred) for pred in predictions_list)
    images_with_gt = sum(1 for gt in ground_truth_list if len(gt) > 0)
    images_with_pred = sum(1 for pred in predictions_list if len(pred) > 0)
    
    print(f"\nDiagnostics:")
    print(f"  Total images evaluated: {len(predictions_list)}")
    print(f"  Images with GT boxes: {images_with_gt} ({100*images_with_gt/len(predictions_list):.1f}%)")
    print(f"  Images with predictions: {images_with_pred} ({100*images_with_pred/len(predictions_list):.1f}%)")
    print(f"  Total GT boxes: {total_gt_boxes}")
    print(f"  Total predicted boxes: {total_pred_boxes}")
    print(f"  Avg GT boxes per image: {total_gt_boxes/len(predictions_list):.2f}")
    print(f"  Avg pred boxes per image: {total_pred_boxes/len(predictions_list):.2f}")
    
    # Check objectness scores
    if len(pred_grids_all) > 0:
        max_obj_scores = []
        for batch_grids in pred_grids_all[:2]:  # Check first 2 batches
            for img_idx in range(batch_grids.shape[0]):
                max_obj = np.max(batch_grids[img_idx, ..., 0])
                max_obj_scores.append(max_obj)
                if len(max_obj_scores) >= 10:
                    break
            if len(max_obj_scores) >= 10:
                break
        if max_obj_scores:
            print(f"  Max objectness scores (first {len(max_obj_scores)} images): {[f'{s:.3f}' for s in max_obj_scores]}")
            print(f"  Avg max objectness: {np.mean(max_obj_scores):.3f}")
    
    # Detailed matching analysis before computing mAP
    print(f"\n{'='*60}")
    print("Matching Analysis:")
    print(f"{'='*60}")
    
    # Analyze class distribution
    gt_class_counts = {}
    pred_class_counts = {}
    for gts in ground_truth_list:
        for gt in gts:
            cls = gt.get("class_id", 0)
            gt_class_counts[cls] = gt_class_counts.get(cls, 0) + 1
    for preds in predictions_list:
        for pred in preds:
            cls = pred.get("class_id", 0)
            pred_class_counts[cls] = pred_class_counts.get(cls, 0) + 1
    
    print(f"\nClass Distribution:")
    print(f"  GT classes: {sorted(gt_class_counts.items())}")
    print(f"  Pred classes: {sorted(pred_class_counts.items())}")
    
    # Analyze IoU distribution for best matches
    all_ious = []
    matched_predictions = 0
    matched_gts = set()
    
    for img_idx, (preds, gts) in enumerate(zip(predictions_list, ground_truth_list)):
        for pred in preds:
            best_iou = 0.0
            best_gt_idx = None
            for gt_idx, gt in enumerate(gts):
                # Check class match first
                if pred.get("class_id") == gt.get("class_id"):
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = (img_idx, gt_idx)
            if best_iou > 0:
                all_ious.append(best_iou)
                if best_iou >= 0.5:
                    matched_predictions += 1
                    if best_gt_idx:
                        matched_gts.add(best_gt_idx)
    
    if all_ious:
        print(f"\nIoU Analysis (best matches per prediction):")
        print(f"  Total predictions analyzed: {sum(len(p) for p in predictions_list)}")
        print(f"  Predictions with IoU > 0: {len(all_ious)}")
        print(f"  Predictions with IoU >= 0.5: {matched_predictions}")
        print(f"  Unique GT boxes matched: {len(matched_gts)}")
        print(f"  IoU stats: min={np.min(all_ious):.3f}, max={np.max(all_ious):.3f}, mean={np.mean(all_ious):.3f}, median={np.median(all_ious):.3f}")
        print(f"  IoU distribution:")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(all_ious, bins=bins)
        for i in range(len(hist)):
            print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]}")
    else:
        print(f"\n  [Warning] No IoU matches found between predictions and GT!")
        print(f"  This suggests class mismatches or box location issues.")
    
    # Box coordinate analysis - check if boxes are in wrong format
    print(f"\n{'='*60}")
    print("Box Coordinate Analysis:")
    print(f"{'='*60}")
    
    # Sample a few predictions and GT boxes to compare
    sample_preds = []
    sample_gts = []
    for img_idx in range(min(5, len(predictions_list))):
        if len(predictions_list[img_idx]) > 0 and len(ground_truth_list[img_idx]) > 0:
            sample_preds.append((img_idx, predictions_list[img_idx][0]))  # First prediction
            sample_gts.append((img_idx, ground_truth_list[img_idx][0]))  # First GT
    
    if sample_preds and sample_gts:
        print(f"\nSample Box Coordinates (first prediction and GT per image):")
        for (img_idx, pred), (gt_img_idx, gt) in zip(sample_preds[:3], sample_gts[:3]):
            if img_idx == gt_img_idx:
                pred_bbox = pred["bbox"]
                gt_bbox = gt["bbox"]
                print(f"\n  Image {img_idx}:")
                print(f"    Pred: class={pred['class_id']}, bbox={pred_bbox}, score={pred['score']:.3f}")
                print(f"    GT:   class={gt['class_id']}, bbox={gt_bbox}")
                print(f"    Format: [ymin, xmin, ymax, xmax] (normalized 0-1)")
                # Check if boxes are valid
                pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
                gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                print(f"    Pred area: {pred_area:.4f}, GT area: {gt_area:.4f}")
                iou = compute_iou(pred_bbox, gt_bbox)
                print(f"    IoU: {iou:.4f}")
    
    # Check box size distribution
    pred_areas = []
    gt_areas = []
    for preds in predictions_list:
        for pred in preds:
            bbox = pred["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            pred_areas.append(area)
    for gts in ground_truth_list:
        for gt in gts:
            bbox = gt["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            gt_areas.append(area)
    
    if pred_areas and gt_areas:
        print(f"\nBox Size Distribution:")
        print(f"  Pred areas: min={np.min(pred_areas):.4f}, max={np.max(pred_areas):.4f}, mean={np.mean(pred_areas):.4f}, median={np.median(pred_areas):.4f}")
        print(f"  GT areas:   min={np.min(gt_areas):.4f}, max={np.max(gt_areas):.4f}, mean={np.mean(gt_areas):.4f}, median={np.median(gt_areas):.4f}")
        print(f"  [Note] If pred areas are much smaller/larger than GT, bbox regression may be wrong.")
    
    # Compute mAP
    print(f"\nEvaluated {len(predictions_list)} images")
    map_3 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.3)
    map_5 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.5)
    map_8 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.8)
    
    print(f"\nResults:")
    print(f"Mean Average Precision at IoU threshold 0.3: {map_3:.4f}")
    print(f"Mean Average Precision at IoU threshold 0.5: {map_5:.4f}")
    print(f"Mean Average Precision at IoU threshold 0.8: {map_8:.4f}")
    
    # Print some example predictions
    print(f"\nExample predictions (first 5 images):")
    for i in range(min(5, len(predictions_list))):
        print(f"\nImage {i+1}:")
        print(f"  GT boxes: {len(ground_truth_list[i])}")
        print(f"  Pred boxes: {len(predictions_list[i])}")
        if predictions_list[i]:
            print(f"  Top prediction: class={predictions_list[i][0]['class_id']}, score={predictions_list[i][0]['score']:.3f}")
        if ground_truth_list[i]:
            print(f"  GT classes: {[gt['class_id'] for gt in ground_truth_list[i]]}")
    """
    # Print some example predictions
    print(f"\nExample predictions (first 5 images):")
    for i in range(min(5, len(predictions_list))):
        print(f"\nImage {i+1}:")
        print(f"  GT boxes: {len(ground_truth_list[i])}")
        print(f"  Pred boxes: {len(predictions_list[i])}")
        if predictions_list[i]:
            print(f"  Top prediction: class={predictions_list[i][0]['class_id']}, score={predictions_list[i][0]['score']:.3f}")
    
    # Visualize first 5 images with predicted boxes
    print("\n" + "=" * 60)
    print("Visualizing predictions...")
    print("=" * 60)
    """
    # Find images with predictions
    images_with_predictions = []
    for i, pred_boxes in enumerate(ground_truth_list):
        if len(pred_boxes) > 0:
            images_with_predictions.append(i)
    
    if len(images_with_predictions) == 0:
        print("\nNo images with predicted boxes found. Model is not detecting any objects.")
    else:
        print(f"\nShowing first 5 examples of with ground truth boxes:")
        for idx, img_idx in enumerate(images_with_predictions[:5]):
            print(f"\nVisualizing image {img_idx + 1}")
            draw_boxes(
                images_list[img_idx],
                ground_truth_list[img_idx],
                predictions_list[img_idx],
                class_names
            )


if __name__ == "__main__":
    main()

