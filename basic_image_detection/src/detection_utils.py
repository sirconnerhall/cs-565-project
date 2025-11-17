"""
Utilities for object detection: NMS, format conversion, evaluation metrics.
"""

import numpy as np
import tensorflow as tf


def compute_iou(box1, box2):
    """
    Compute IoU (Intersection over Union) between two boxes.
    
    Args:
        box1, box2: Boxes in [ymin, xmin, ymax, xmax] format (normalized 0-1)
    
    Returns:
        IoU value (float)
    """
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
        boxes: List of dicts with "bbox" key, or list of [ymin, xmin, ymax, xmax]
        scores: List of scores, or boxes can have "score" key
        iou_threshold: IoU threshold for suppression
        max_output_size: Maximum number of boxes to keep
    
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Extract scores if boxes are dicts
    if isinstance(boxes[0], dict):
        box_list = [b["bbox"] for b in boxes]
        score_list = [b.get("score", 0.0) for b in boxes] if scores is None else scores
    else:
        box_list = boxes
        score_list = scores if scores is not None else [1.0] * len(boxes)
    
    # Sort by score (descending)
    indices = sorted(range(len(box_list)), key=lambda i: score_list[i], reverse=True)
    
    keep = []
    while len(indices) > 0 and len(keep) < max_output_size:
        # Take highest scoring box
        current_idx = indices[0]
        keep.append(current_idx)
        indices = indices[1:]
        
        # Remove boxes with high IoU
        current_box = box_list[current_idx]
        remaining_indices = []
        for idx in indices:
            other_box = box_list[idx]
            iou = compute_iou(current_box, other_box)
            if iou < iou_threshold:
                remaining_indices.append(idx)
        indices = remaining_indices
    
    return keep


def decode_predictions_grid(
    grid_pred,
    num_classes,
    threshold=0.5,
    nms_iou=0.5,
    max_boxes=20,
    min_box_size=0.01,
):
    """
    Decode grid predictions to bounding boxes with NMS.
    
    Input format: [H, W, 1 + 4 + num_classes] where:
    - [..., 0]: objectness
    - [..., 1:5]: bbox (cx, cy, w, h) normalized
    - [..., 5:]: class probabilities
    
    Output: List of dicts with bbox, class_id, score
    
    Args:
        grid_pred: [H, W, 5 + num_classes] numpy array
        num_classes: Number of classes
        threshold: Objectness threshold
        nms_iou: IoU threshold for NMS
        max_boxes: Maximum boxes to return
        min_box_size: Minimum box size (normalized) to filter out
    
    Returns:
        List of detection dicts: [{"bbox": [ymin, xmin, ymax, xmax], "class_id": int, "score": float}, ...]
    """
    H, W = grid_pred.shape[:2]
    boxes = []
    
    for y in range(H):
        for x in range(W):
            cell = grid_pred[y, x]
            
            obj = cell[0]
            if obj < threshold:
                continue
            
            # Bounding box (cx, cy, w, h in normalized coordinates)
            cx, cy, w, h = cell[1:5]
            
            # Filter out very small boxes
            if w < min_box_size or h < min_box_size:
                continue
            
            # Convert from center format to corner format
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


def compute_map(
    predictions_list,
    ground_truth_list,
    num_classes,
    iou_threshold=0.5,
):
    """
    Compute mean Average Precision (mAP) at a given IoU threshold.
    
    Args:
        predictions_list: List of lists, each containing prediction dicts for one image
        ground_truth_list: List of lists, each containing GT dicts for one image
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching predictions to GT
    
    Returns:
        mAP value (float)
    """
    # Per-class AP
    ap_per_class = []
    
    for class_id in range(num_classes):
        # Collect all predictions and GT for this class
        pred_scores = []
        pred_matched = []
        gt_count = 0
        
        for preds, gts in zip(predictions_list, ground_truth_list):
            # Count GT boxes for this class
            class_gts = [gt for gt in gts if gt.get("class_id") == class_id]
            gt_count += len(class_gts)
            
            # Get predictions for this class
            class_preds = [p for p in preds if p.get("class_id") == class_id]
            
            # Match predictions to GT
            matched_gt = set()
            for pred in class_preds:
                pred_scores.append(pred["score"])
                matched = False
                
                for i, gt in enumerate(class_gts):
                    if i in matched_gt:
                        continue
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou >= iou_threshold:
                        matched = True
                        matched_gt.add(i)
                        break
                
                pred_matched.append(matched)
        
        if gt_count == 0:
            continue  # No GT for this class, skip
        
        # Sort predictions by score
        sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        sorted_matched = [pred_matched[i] for i in sorted_indices]
        
        # Compute precision and recall
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        for matched in sorted_matched:
            if matched:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / gt_count if gt_count > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            # Find max precision at recall >= r
            max_prec = 0.0
            for rec, prec in zip(recalls, precisions):
                if rec >= r:
                    max_prec = max(max_prec, prec)
            ap += max_prec
        ap /= 11.0
        
        ap_per_class.append(ap)
    
    # mAP is average of per-class AP
    map_value = np.mean(ap_per_class) if len(ap_per_class) > 0 else 0.0
    return map_value


def convert_bbox_format(bbox, from_format="cxcywh", to_format="xyxy"):
    """
    Convert bounding box between formats.
    
    Args:
        bbox: [4] array with box coordinates
        from_format: "cxcywh" (center x, center y, width, height) or "xyxy" (xmin, ymin, xmax, ymax)
        to_format: "cxcywh" or "xyxy"
    
    Returns:
        Converted bbox array
    """
    if from_format == to_format:
        return bbox
    
    if from_format == "cxcywh" and to_format == "xyxy":
        cx, cy, w, h = bbox
        xmin = cx - w / 2.0
        ymin = cy - h / 2.0
        xmax = cx + w / 2.0
        ymax = cy + h / 2.0
        return np.array([xmin, ymin, xmax, ymax])
    
    elif from_format == "xyxy" and to_format == "cxcywh":
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        return np.array([cx, cy, w, h])
    
    else:
        raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")

