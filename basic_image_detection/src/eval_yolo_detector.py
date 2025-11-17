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

from cct_pipeline import make_cct_dataset, load_cct_annotations, extract_cct_metadata_features
from cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from build_yolo_detector import build_ssd_detector_with_metadata
from train_cct_multimodal_detector import DetectionLossFocal, objectness_accuracy, make_component_loss_metrics
from detection_utils import decode_predictions_grid, compute_map
from cct_splits_utils import get_filelist_from_splits_or_config


def load_config():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "coco_multilabel_config.json"
    
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
    
    model_name = config.get("detector_model_name", "yolo_detector")
    models_dir = project_root / config["models_dir"]
    model_path = models_dir / f"{model_name}_best.keras"
    
    pretrained_model_type = config.get("pretrained_model_type", "ssd_mobilenet_v2")
    
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 4.0)
    
    print("Loading model:", model_path)
    
    loss_fn = DetectionLossFocal(
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        positive_weight=positive_weight
    )
    
    # Create component loss metrics (needed for loading model that was saved with these metrics)
    component_metrics = make_component_loss_metrics(loss_fn)
    
    # Build custom_objects dict with all metrics
    custom_objects = {
        "DetectionLossFocal": DetectionLossFocal,
        "objectness_accuracy": objectness_accuracy,
    }
    
    # Add component loss metrics
    for metric_fn in component_metrics:
        custom_objects[metric_fn.__name__] = metric_fn
    
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
    )
    
    model.summary()
    
    # Load dataset
    dataset_name = config.get("dataset", "cct")
    metadata_dim = 5
    
    if dataset_name != "cct":
        raise ValueError(f"YOLO detector currently only supports CCT dataset")
    
    # Load validation samples for metadata
    val_filelist = get_filelist_from_splits_or_config(config, "val", config["cct_annotations"])
    
    samples_val, _ = load_cct_annotations(
        metadata_path=config["cct_annotations"],
        bboxes_path=config["cct_bb_annotations"],
        images_root=config["cct_images_root"],
        filelist_path=val_filelist,
        filter_empty=False,  # Keep all for evaluation
    )
    
    use_tfrecords = config.get("cct_use_tfrecords", True)
    cct_tfrecords_dir = config.get("cct_tfrecords_dir")
    
    if use_tfrecords and cct_tfrecords_dir and Path(cct_tfrecords_dir).exists():
        val_ds_raw, val_info = make_cct_tfrecords_dataset(
            tfrecords_dir=cct_tfrecords_dir,
            split="val",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
        )
    else:
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
    
    num_classes = val_info.features["objects"]["label"].num_classes
    class_names = val_info.features["objects"]["label"].names
    
    # Add metadata
    if not use_tfrecords:
        val_metadata_array = np.array([extract_cct_metadata_features(s) for s in samples_val], dtype=np.float32)
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
        
        val_ds_with_meta = val_ds_zipped.map(combine_metadata, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds_with_meta = val_ds_with_meta.batch(batch_size)
    else:
        def add_placeholder_metadata(images, targets):
            batch_size = tf.shape(images)[0]
            brightness = tf.reduce_mean(images, axis=[1, 2, 3])
            location_id = tf.zeros([batch_size], dtype=tf.float32)
            hour = tf.ones([batch_size], dtype=tf.float32) * 0.5
            day_of_week = tf.ones([batch_size], dtype=tf.float32) * 0.5
            month = tf.ones([batch_size], dtype=tf.float32) * 0.5
            metadata = tf.stack([location_id, hour, day_of_week, month, brightness], axis=1)
            return (images, metadata), targets
        
        val_ds_with_meta = val_ds_raw.map(add_placeholder_metadata, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating model...")
    print("=" * 60)
    
    # Collect predictions, ground truth, and images for visualization
    predictions_list = []
    ground_truth_list = []
    images_list = []  # Store images for visualization
    
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
        
        # Get predictions
        if metadata is not None:
            pred_grids = model([images, metadata], training=False)
        else:
            pred_grids = model(images, training=False)
        
        # Convert to numpy
        pred_grids_np = pred_grids.numpy()
        images_np = images.numpy()
        
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
            pred_boxes = decode_predictions_grid(
                pred_grid,
                num_classes=num_classes,
                threshold=0.5,
                nms_iou=0.5,
                max_boxes=20,
            )
            predictions_list.append(pred_boxes)
            
            # Get ground truth boxes
            gt_boxes = []
            if isinstance(targets, dict):
                bboxes = targets["bboxes"][i].numpy() if hasattr(targets["bboxes"], "numpy") else targets["bboxes"][i]
                labels = targets["labels"][i].numpy() if hasattr(targets["labels"], "numpy") else targets["labels"][i]
                
                for bbox, label in zip(bboxes, labels):
                    if np.sum(np.abs(bbox)) > 0:  # Valid bbox
                        gt_boxes.append({
                            "bbox": bbox.tolist(),  # [ymin, xmin, ymax, xmax]
                            "class_id": int(label),
                        })
            
            ground_truth_list.append(gt_boxes)
            images_list.append(image_np)  # Store image for visualization
        
        # Limit evaluation to first few batches for speed
        if batch_idx >= 10:
            break
    
    # Compute mAP
    print(f"\nEvaluated {len(predictions_list)} images")
    map_50 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.5)
    map_75 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.75)
    
    print(f"\nResults:")
    print(f"  mAP@0.5: {map_50:.4f}")
    print(f"  mAP@0.75: {map_75:.4f}")
    
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
    
    # Find images with predictions
    images_with_predictions = []
    for i, pred_boxes in enumerate(predictions_list):
        if len(pred_boxes) > 0:
            images_with_predictions.append(i)
    
    if len(images_with_predictions) == 0:
        print("\nNo images with predicted boxes found. Model is not detecting any objects.")
    else:
        print(f"\nFound {len(images_with_predictions)} images with predictions. Showing first 5:")
        for idx, img_idx in enumerate(images_with_predictions[:5]):
            print(f"\nVisualizing image {img_idx + 1} (has {len(predictions_list[img_idx])} predicted boxes)")
            draw_boxes(
                images_list[img_idx],
                ground_truth_list[img_idx],
                predictions_list[img_idx],
                class_names
            )


if __name__ == "__main__":
    main()

