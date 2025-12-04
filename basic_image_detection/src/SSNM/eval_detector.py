"""
Evaluate SSNM detector (Single Stage, No Metadata).
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..pipelines.coco_tfds_pipeline import make_coco_dataset
from ..pipelines.cct_tfrecords_pipeline import make_cct_tfrecords_dataset
from ..utils.detection_utils import compute_map, DetectionLossFocal, make_component_loss_metrics, objectness_accuracy
from .train_detector import infer_grid_size, make_grid_encoder


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "config.json"
    with open(config_path, "r") as f:
        return json.load(f), project_root


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [ymin, xmin, ymax, xmax] format."""
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    
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
    """Non-maximum suppression to remove overlapping boxes."""
    if len(boxes) == 0:
        return []
    
    indices = sorted(range(len(boxes)), key=lambda i: boxes[i]["score"], reverse=True)
    keep = []
    
    while len(indices) > 0 and len(keep) < max_output_size:
        current_idx = indices[0]
        keep.append(current_idx)
        indices = indices[1:]
        
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
    """Decode grid predictions to bounding boxes with NMS."""
    S = grid_pred.shape[0]
    boxes = []

    for gy in range(S):
        for gx in range(S):
            cell = grid_pred[gy, gx]
            obj = cell[0]
            if obj < threshold:
                continue

            cx, cy, w, h = cell[1:5]
            
            if w < min_box_size or h < min_box_size:
                continue

            xmin = cx - w / 2.0
            ymin = cy - h / 2.0
            xmax = cx + w / 2.0
            ymax = cy + h / 2.0

            xmin = np.clip(xmin, 0, 1)
            ymin = np.clip(ymin, 0, 1)
            xmax = np.clip(xmax, 0, 1)
            ymax = np.clip(ymax, 0, 1)
            
            if (xmax - xmin) < min_box_size or (ymax - ymin) < min_box_size:
                continue

            class_probs = cell[5:]
            class_id = int(np.argmax(class_probs))
            score = float(obj * class_probs[class_id])

            boxes.append({
                "bbox": [ymin, xmin, ymax, xmax],
                "class_id": class_id,
                "score": score,
            })
    
    if len(boxes) > 0:
        keep_indices = nms(boxes, None, iou_threshold=nms_iou, max_output_size=max_boxes)
        boxes = [boxes[i] for i in keep_indices]

    return boxes


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
    
    # Model name is set here, not from config
    model_name = "SSNM"
    
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    models_dir = project_root / config["models_dir"]
    model_path = models_dir / f"{model_name}_best.keras"

    print("Loading model:", model_path)
    
    focal_gamma = config.get("focal_gamma", 2.0)
    focal_alpha = config.get("focal_alpha", 0.5)
    positive_weight = config.get("positive_weight", 5.0)
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

    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
    )

    model.summary()

    # Load dataset
    dataset_name = config.get("dataset", "cct")
    
    if dataset_name == "coco":
        val_split = config.get("val_split", "validation[:20%]")
        val_ds_raw, val_info = make_coco_dataset(
            split=val_split,
            batch_size=batch_size,
            image_size=image_size,
        )
    elif dataset_name == "cct":
        cct_tfrecords_dir = config.get("cct_tfrecords_dir")
        if not cct_tfrecords_dir or not Path(cct_tfrecords_dir).exists():
            raise ValueError(f"TFRecords directory not found: {cct_tfrecords_dir}.")
        
        print(f"[Dataset] Using TFRecords from {cct_tfrecords_dir}")
        val_ds_raw, val_info = make_cct_tfrecords_dataset(
            tfrecords_dir=cct_tfrecords_dir,
            split="val",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    num_classes = val_info.features["objects"]["label"].num_classes
    class_names = val_info.features["objects"]["label"].names
    grid_size = infer_grid_size(image_size)

    encoder = make_grid_encoder(num_classes, grid_size)
    AUTOTUNE = tf.data.AUTOTUNE
    val_ds = val_ds_raw.map(encoder).prefetch(AUTOTUNE)

    print("\n=== Running numeric evaluation ===")
    result = model.evaluate(val_ds, return_dict=True)
    print(result)

    print("\n" + "=" * 60)
    print("Evaluating model...")
    print("=" * 60)
    
    predictions_list = []
    ground_truth_list = []
    images_list = []
    
    DETECTION_THRESHOLD = 0.1
    
    for batch_idx, batch in enumerate(val_ds):
        if isinstance(batch, tuple) and len(batch) == 2:
            images, grid_true = batch
        else:
            continue
        
        if images.dtype == tf.float32 and tf.reduce_max(images) <= 1.0:
            images_for_model = images * 255.0
        else:
            images_for_model = images
        
        pred_grids = model(images_for_model, training=False)
        pred_grids_np = pred_grids.numpy()
        images_np = images.numpy()
        grid_true_np = grid_true.numpy()
        
        batch_size_actual = pred_grids_np.shape[0]
        for i in range(batch_size_actual):
            pred_grid = pred_grids_np[i]
            image_np = images_np[i]
            grid_t = grid_true_np[i]
            
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            
            pred_boxes = decode_predictions(
                pred_grid,
                num_classes=num_classes,
                threshold=DETECTION_THRESHOLD,
                nms_iou=0.5,
                max_boxes=20,
            )
            predictions_list.append(pred_boxes)
            
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
            
            ground_truth_list.append(gt_boxes)
            images_list.append(image_np)
        
        if batch_idx >= 10:
            break
    
    print(f"\nEvaluated {len(predictions_list)} images")
    map_3 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.3)
    map_5 = compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.5)
    
    print(f"\nResults:")
    print(f"  mAP@0.3: {map_3:.4f}")
    print(f"  mAP@0.5: {map_5:.4f}")


if __name__ == "__main__":
    main()

