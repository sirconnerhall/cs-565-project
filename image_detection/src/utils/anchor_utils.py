"""
Anchor box utilities for YOLO-style object detection.

Handles anchor box generation, matching, and encoding/decoding.
"""

import numpy as np
import tensorflow as tf


def generate_default_anchors(grid_size, num_anchors=3, anchor_scales=None):
    """
    Generate default anchor boxes for a grid.
    
    Args:
        grid_size: (height, width) of the grid
        num_anchors: Number of anchors per grid cell (default 3)
        anchor_scales: List of (width, height) anchor scales in normalized coordinates.
                      If None, uses standard YOLO anchors.
    
    Returns:
        anchors: Array of shape [grid_h, grid_w, num_anchors, 4] where last dim is [cx, cy, w, h]
                All coordinates are normalized [0, 1]
    """
    grid_h, grid_w = grid_size
    
    # Default YOLO anchors (normalized widths and heights)
    # These are typical for COCO dataset, adjust for your dataset
    if anchor_scales is None:
        # Standard YOLO anchors (width, height) - adjust based on your dataset
        anchor_scales = [
            (0.1, 0.1),   # Small objects
            (0.3, 0.3),   # Medium objects
            (0.5, 0.5),   # Large objects
        ]
        # If num_anchors > 3, add more scales
        if num_anchors > 3:
            anchor_scales.extend([
                (0.2, 0.4),   # Tall objects
                (0.4, 0.2),   # Wide objects
            ][:num_anchors - 3])
    
    anchors = np.zeros((grid_h, grid_w, num_anchors, 4), dtype=np.float32)
    
    for y in range(grid_h):
        for x in range(grid_w):
            # Center of grid cell (normalized)
            cx = (x + 0.5) / grid_w
            cy = (y + 0.5) / grid_h
            
            for a in range(num_anchors):
                w, h = anchor_scales[a]
                anchors[y, x, a] = [cx, cy, w, h]
    
    return anchors


def compute_anchor_iou(anchor, box):
    """
    Compute IoU between an anchor box and a ground truth box.
    
    Args:
        anchor: [cx, cy, w, h] in normalized coordinates
        box: [ymin, xmin, ymax, xmax] in normalized coordinates
    
    Returns:
        IoU value
    """
    # Convert anchor to corner format
    cx, cy, w, h = anchor
    anchor_xmin = cx - w / 2.0
    anchor_ymin = cy - h / 2.0
    anchor_xmax = cx + w / 2.0
    anchor_ymax = cy + h / 2.0
    
    # Convert box to corner format (already in corner format)
    box_ymin, box_xmin, box_ymax, box_xmax = box
    
    # Compute intersection
    inter_xmin = max(anchor_xmin, box_xmin)
    inter_ymin = max(anchor_ymin, box_ymin)
    inter_xmax = min(anchor_xmax, box_xmax)
    inter_ymax = min(anchor_ymax, box_ymax)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    anchor_area = w * h
    box_area = (box_ymax - box_ymin) * (box_xmax - box_xmin)
    union_area = anchor_area + box_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_box_to_anchor(box, anchors, iou_threshold=0.5):
    """
    Match a ground truth box to the best anchor box.
    
    Args:
        box: [ymin, xmin, ymax, xmax] in normalized coordinates
        anchors: Array of shape [grid_h, grid_w, num_anchors, 4] or [num_anchors, 4]
        iou_threshold: Minimum IoU to consider a match
    
    Returns:
        (grid_y, grid_x, anchor_idx, iou) or None if no match
    """
    if anchors.ndim == 4:
        grid_h, grid_w, num_anchors, _ = anchors.shape
        best_iou = 0.0
        best_match = None
        
        for y in range(grid_h):
            for x in range(grid_w):
                for a in range(num_anchors):
                    anchor = anchors[y, x, a]
                    iou = compute_anchor_iou(anchor, box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = (y, x, a, iou)
        
        if best_iou >= iou_threshold:
            return best_match
        return None
    else:
        # Single anchor set
        num_anchors = anchors.shape[0]
        best_iou = 0.0
        best_anchor_idx = None
        
        for a in range(num_anchors):
            iou = compute_anchor_iou(anchors[a], box)
            if iou > best_iou:
                best_iou = iou
                best_anchor_idx = a
        
        if best_iou >= iou_threshold:
            return (None, None, best_anchor_idx, best_iou)
        return None


def encode_box_to_anchor(box, anchor):
    """
    Encode a ground truth box as offsets from an anchor box.
    
    Args:
        box: [ymin, xmin, ymax, xmax] in normalized coordinates
        anchor: [cx, cy, w, h] in normalized coordinates
    
    Returns:
        [offset_x, offset_y, offset_w, offset_h]
    """
    # Convert box to center format
    box_ymin, box_xmin, box_ymax, box_xmax = box
    box_cx = (box_xmin + box_xmax) / 2.0
    box_cy = (box_ymin + box_ymax) / 2.0
    box_w = box_xmax - box_xmin
    box_h = box_ymax - box_ymin
    
    anchor_cx, anchor_cy, anchor_w, anchor_h = anchor
    
    # Compute offsets (YOLO-style encoding)
    # Offset for center (relative to grid cell, then normalized by anchor size)
    offset_x = (box_cx - anchor_cx) / anchor_w if anchor_w > 0 else 0.0
    offset_y = (box_cy - anchor_cy) / anchor_h if anchor_h > 0 else 0.0
    
    # Offset for size (log space)
    offset_w = np.log(box_w / anchor_w) if anchor_w > 0 and box_w > 0 else 0.0
    offset_h = np.log(box_h / anchor_h) if anchor_h > 0 and box_h > 0 else 0.0
    
    return np.array([offset_x, offset_y, offset_w, offset_h], dtype=np.float32)


def decode_anchor_to_box(anchor, offsets):
    """
    Decode anchor offsets back to absolute box coordinates.
    
    Args:
        anchor: [cx, cy, w, h] in normalized coordinates
        offsets: [offset_x, offset_y, offset_w, offset_h]
    
    Returns:
        [ymin, xmin, ymax, xmax] in normalized coordinates
    """
    anchor_cx, anchor_cy, anchor_w, anchor_h = anchor
    offset_x, offset_y, offset_w, offset_h = offsets
    
    # Decode center
    box_cx = anchor_cx + offset_x * anchor_w
    box_cy = anchor_cy + offset_y * anchor_h
    
    # Decode size (from log space)
    box_w = anchor_w * np.exp(offset_w)
    box_h = anchor_h * np.exp(offset_h)
    
    # Convert to corner format
    xmin = box_cx - box_w / 2.0
    ymin = box_cy - box_h / 2.0
    xmax = box_cx + box_w / 2.0
    ymax = box_cy + box_h / 2.0
    
    return np.array([ymin, xmin, ymax, xmax], dtype=np.float32)


def compute_anchors_from_dataset(bboxes_list, num_anchors=3, grid_size=(7, 7)):
    """
    Compute optimal anchor boxes using k-means clustering on dataset.
    
    Args:
        bboxes_list: List of bbox arrays, each bbox is [ymin, xmin, ymax, xmax]
        num_anchors: Number of anchors to generate
        grid_size: Grid size (not used for clustering, but for reference)
    
    Returns:
        List of (width, height) anchor scales in normalized coordinates
    """
    # Extract all box widths and heights
    widths = []
    heights = []
    
    for bboxes in bboxes_list:
        for bbox in bboxes:
            if len(bbox) == 4:
                ymin, xmin, ymax, xmax = bbox
                w = xmax - xmin
                h = ymax - ymin
                if w > 0 and h > 0:
                    widths.append(w)
                    heights.append(h)
    
    if len(widths) == 0:
        # Fall back to default anchors
        return None  # Will use defaults in generate_default_anchors
    
    # Simple k-means on box sizes (if sklearn available)
    try:
        from sklearn.cluster import KMeans
        
        box_sizes = np.array([[w, h] for w, h in zip(widths, heights)])
        kmeans = KMeans(n_clusters=num_anchors, random_state=42, n_init=10)
        kmeans.fit(box_sizes)
        
        anchors = kmeans.cluster_centers_
        # Sort by area
        areas = anchors[:, 0] * anchors[:, 1]
        sorted_indices = np.argsort(areas)
        anchors = anchors[sorted_indices]
        
        # Convert to list of tuples
        anchor_scales = [(float(w), float(h)) for w, h in anchors]
        
        return anchor_scales
    except ImportError:
        # sklearn not available, return None to use defaults
        print("[Warning] sklearn not available, using default anchor scales")
        return None


if __name__ == "__main__":
    # Test anchor utilities
    print("Testing anchor utilities...")
    
    # Generate default anchors
    anchors = generate_default_anchors((7, 7), num_anchors=3)
    print(f"Anchors shape: {anchors.shape}")
    print(f"Sample anchor at (0,0,0): {anchors[0, 0, 0]}")
    
    # Test matching
    test_box = [0.3, 0.3, 0.5, 0.5]  # [ymin, xmin, ymax, xmax]
    match = match_box_to_anchor(test_box, anchors)
    if match:
        y, x, a, iou = match
        print(f"Matched box to anchor at grid ({y}, {x}), anchor {a}, IoU: {iou:.3f}")
    
    # Test encoding/decoding
    anchor = anchors[0, 0, 0]
    offsets = encode_box_to_anchor(test_box, anchor)
    decoded_box = decode_anchor_to_box(anchor, offsets)
    print(f"Original box: {test_box}")
    print(f"Decoded box: {decoded_box}")
    print(f"Difference: {np.abs(np.array(test_box) - decoded_box)}")

