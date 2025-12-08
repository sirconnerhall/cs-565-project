"""
Evaluate SSM (Single Stage, with Metadata) detection model.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..pipelines.cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from .build_detector import build_detector
from ..utils.detection_utils import (
    decode_predictions_anchors,
    compute_map,
    compute_iou,
    DetectionLossFocal,
    objectness_accuracy,
    make_component_loss_metrics,
)


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
    model_name = "SSM"
    models_dir = project_root / config["models_dir"]
    model_path = models_dir / f"{model_name}_best.keras"
    
    pretrained_model_type = config.get("pretrained_model_type", "efficientnet_b0")
    num_anchors = config.get("num_anchors", 3)
    use_anchors = config.get("use_anchors", True)
    
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 4.0)
    bbox_loss_weight = config.get("bbox_loss_weight", 10.0)
    
    print("=" * 60)
    print(f"Evaluating {model_name} Model")
    print("=" * 60)
    print(f"Loading model: {model_path}")
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first.")
        return
    
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
    # Load dataset
    dataset_name = config.get("dataset", "cct")
    metadata_dim = 8  # Updated: 8 features with cyclical encoding
    
    if dataset_name != "cct":
        raise ValueError(f"SSM detector currently only supports CCT dataset")
    
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
        model = build_detector(
            image_size=image_size,
            num_classes=num_classes,
            metadata_dim=metadata_dim,
            backbone_type=pretrained_model_type,
            num_anchors=num_anchors,
        )
        # Check for weights file with the pattern used by SafeModelCheckpoint
        # Training saves as: SSM_best_weights.h5 (replaces .keras with _weights.h5)
        weights_path = model_path.parent / f"{model_name}_best_weights.h5"
        if not weights_path.exists():
            # Fallback: try the simple .h5 suffix
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
    images_list = []
    
    # Infer grid size from image size
    grid_h = image_size[0] // 32
    grid_w = image_size[1] // 32
    grid_size = (grid_h, grid_w)
    
    print(f"Grid size: {grid_h}x{grid_w}")
    print(f"Using anchor-based decoding: {use_anchors}")
    print(f"Number of anchors: {num_anchors}")
    
    max_batches = config.get("eval_max_batches", 50)  # Limit evaluation batches
    print(f"Evaluating on first {max_batches} batches...")
    
    for batch_idx, batch in enumerate(val_ds_with_meta):
        if batch_idx >= max_batches:
            break
            
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
        # Model expects [0, 255] range for preprocessing
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
        
        # Decode predictions for each image in batch
        batch_size_actual = pred_grids_np.shape[0]
        for i in range(batch_size_actual):
            pred_grid = pred_grids_np[i]  # [H, W, num_anchors * (1 + 4 + num_classes)]
            image_np = images_np[i]  # [H, W, 3]
            
            # Convert image from normalized [0, 1] to [0, 255] uint8 if needed
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            # Decode to boxes using anchor-based decoding
            pred_boxes = decode_predictions_anchors(
                pred_grid,
                num_classes=num_classes,
                grid_size=grid_size,
                num_anchors=num_anchors,
                threshold=0.1,  # Objectness threshold
                nms_iou=0.5,  # NMS IoU threshold
                max_boxes=50,  # Maximum boxes per image
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
                    ymin, xmin, ymax, xmax = bbox
                    if (np.sum(np.abs(bbox)) > 1e-6 and  # Not all zeros
                        ymax > ymin and xmax > xmin and  # Valid dimensions
                        ymin >= 0 and xmin >= 0 and ymax <= 1 and xmax <= 1):  # Within bounds
                        gt_boxes.append({
                            "bbox": [float(ymin), float(xmin), float(ymax), float(xmax)],
                            "class_id": int(label),
                        })
            
            ground_truth_list.append(gt_boxes)
            images_list.append(image_np)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1} batches...")
    
    # Diagnostic output
    total_gt_boxes = sum(len(gt) for gt in ground_truth_list)
    total_pred_boxes = sum(len(pred) for pred in predictions_list)
    images_with_gt = sum(1 for gt in ground_truth_list if len(gt) > 0)
    images_with_pred = sum(1 for pred in predictions_list if len(pred) > 0)
    
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total images evaluated: {len(predictions_list)}")
    print(f"Images with GT boxes: {images_with_gt} ({100*images_with_gt/len(predictions_list):.1f}%)")
    print(f"Images with predictions: {images_with_pred} ({100*images_with_pred/len(predictions_list):.1f}%)")
    print(f"Total GT boxes: {total_gt_boxes}")
    print(f"Total predicted boxes: {total_pred_boxes}")
    print(f"Avg GT boxes per image: {total_gt_boxes/len(predictions_list):.2f}")
    print(f"Avg pred boxes per image: {total_pred_boxes/len(predictions_list):.2f}")
    
    # Compute mAP at different IoU thresholds
    print(f"\n{'='*60}")
    print("Computing mAP...")
    print(f"{'='*60}")
    
    map_3 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.3)
    map_5 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.5)
    map_8 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.8)
    
    print(f"\nResults:")
    print(f"  mAP@0.3: {map_3:.4f}")
    print(f"  mAP@0.5: {map_5:.4f}")
    print(f"  mAP@0.8: {map_8:.4f}")
    
    # Print some example predictions
    print(f"\n{'='*60}")
    print("Example Predictions (first 5 images)")
    print(f"{'='*60}")
    for i in range(min(5, len(predictions_list))):
        print(f"\nImage {i+1}:")
        print(f"  GT boxes: {len(ground_truth_list[i])}")
        print(f"  Pred boxes: {len(predictions_list[i])}")
        if predictions_list[i]:
            top_pred = predictions_list[i][0]
            print(f"  Top prediction: class={top_pred['class_id']}, score={top_pred['score']:.3f}, bbox={top_pred['bbox']}")
        if ground_truth_list[i]:
            print(f"  GT classes: {[gt['class_id'] for gt in ground_truth_list[i]]}")
    
    # Optional: Visualize some examples
    visualize = config.get("eval_visualize", False)
    if visualize:
        print(f"\n{'='*60}")
        print("Visualizing predictions...")
        print(f"{'='*60}")
        
        # Find images with both predictions and GT
        images_to_show = []
        for i, (preds, gts) in enumerate(zip(predictions_list, ground_truth_list)):
            if len(preds) > 0 and len(gts) > 0:
                images_to_show.append(i)
            if len(images_to_show) >= 5:
                break
        
        if len(images_to_show) == 0:
            print("No images with both predictions and GT boxes found.")
        else:
            print(f"Showing {len(images_to_show)} examples...")
            for idx in images_to_show:
                print(f"\nVisualizing image {idx + 1}")
                draw_boxes(
                    images_list[idx],
                    ground_truth_list[idx],
                    predictions_list[idx],
                    class_names
                )


if __name__ == "__main__":
    main()

