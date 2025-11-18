"""
Utilities for object detection: NMS, format conversion, evaluation metrics.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


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
    """
    def __init__(self, focal_gamma=2.0, focal_alpha=0.5, positive_weight=5.0, name="detection_loss_focal", **kwargs):
        super().__init__(name=name, **kwargs)
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.positive_weight = positive_weight  # Additional weight for positive examples
    
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
        box_true = y_true[..., 1:5]   # [B, S, S, 4]
        box_pred = y_pred[..., 1:5]   # [B, S, S, 4]
        box_diff = box_true - box_pred
        box_sq = tf.square(box_diff)
        box_loss = tf.reduce_sum(box_sq, axis=-1, keepdims=True)  # [B, S, S, 1]
        box_loss = box_loss * obj_true  # Mask by objectness
        
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


# ----------------------------
# Objectness accuracy metric
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
                    inputs, targets = val_batch
                    if isinstance(inputs, tuple):
                        images, metadata = inputs
                    else:
                        images = inputs
                        metadata = None
                else:
                    return
                
                # Get predictions
                if metadata is not None:
                    predictions = self.model([images, metadata], training=False)
                else:
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
