"""
Anchor-based target encoding for YOLO-style object detection.

Converts ground truth bounding boxes to anchor-based format for training.
"""

import numpy as np
import tensorflow as tf

from .anchor_utils import (
    generate_default_anchors,
    match_box_to_anchor,
    encode_box_to_anchor,
)


def make_anchor_encoder(num_classes, grid_size, num_anchors=3, anchor_scales=None, iou_threshold=0.5):
    """
    Create an anchor-based encoder function.
    
    Args:
        num_classes: Number of object classes
        grid_size: (height, width) of output grid
        num_anchors: Number of anchors per grid cell
        anchor_scales: List of (width, height) anchor scales. If None, uses defaults.
        iou_threshold: Minimum IoU to match a box to an anchor
    
    Returns:
        Encoder function that takes (images, targets) and returns (images, anchor_targets)
    """
    grid_h, grid_w = grid_size
    
    # Generate anchor boxes
    anchors = generate_default_anchors(grid_size, num_anchors, anchor_scales)
    
    # Output depth: objectness + 4 offsets + num_classes per anchor
    depth_per_anchor = 1 + 4 + num_classes
    total_depth = num_anchors * depth_per_anchor
    
    def _tf_fn(images, targets):
        """
        Encode targets to anchor format.
        
        Args:
            images: [B, H, W, 3] image tensor
            targets: Dict with "bboxes" [B, N, 4] and "labels" [B, N]
        
        Returns:
            images, anchor_targets: [B, grid_h, grid_w, total_depth]
        """
        def _np_encode(images_np, bboxes_np, labels_np):
            """
            NumPy-side encoding over the batch.
            """
            B = images_np.shape[0]
            anchor_targets = np.zeros((B, grid_h, grid_w, total_depth), dtype=np.float32)
            
            for b in range(B):
                boxes = bboxes_np[b]  # [max_boxes, 4] in [ymin, xmin, ymax, xmax]
                classes = labels_np[b]  # [max_boxes]
                
                for box, cls in zip(boxes, classes):
                    ymin, xmin, ymax, xmax = box
                    
                    # Skip invalid boxes
                    if ymax <= ymin or xmax <= xmin:
                        continue
                    
                    # Skip padded boxes (all zeros)
                    if np.sum(np.abs(box)) < 1e-6:
                        continue
                    
                    cls_int = int(cls)
                    if cls_int < 0 or cls_int >= num_classes:
                        continue
                    
                    # Match box to best anchor
                    match = match_box_to_anchor(box, anchors, iou_threshold=iou_threshold)
                    
                    if match is None:
                        continue  # No matching anchor found
                    
                    grid_y, grid_x, anchor_idx, iou = match
                    
                    # Get the matched anchor
                    matched_anchor = anchors[grid_y, grid_x, anchor_idx]
                    
                    # Encode box offsets
                    offsets = encode_box_to_anchor(box, matched_anchor)
                    
                    # Compute anchor index in flattened output
                    anchor_start = anchor_idx * depth_per_anchor
                    
                    # Set objectness (use IoU as objectness target, or 1.0 if matched)
                    anchor_targets[b, grid_y, grid_x, anchor_start] = 1.0
                    
                    # Set bbox offsets
                    anchor_targets[b, grid_y, grid_x, anchor_start + 1:anchor_start + 5] = offsets
                    
                    # Set class (one-hot)
                    anchor_targets[b, grid_y, grid_x, anchor_start + 5 + cls_int] = 1.0
            
            return anchor_targets
        
        labels = tf.cast(targets["labels"], tf.int32)
        anchor_targets = tf.numpy_function(
            _np_encode,
            [images, targets["bboxes"], labels],
            tf.float32,
        )
        anchor_targets.set_shape((None, grid_h, grid_w, total_depth))
        return images, anchor_targets
    
    return _tf_fn


def make_anchor_encoder_with_metadata(num_classes, grid_size, num_anchors=3, 
                                      anchor_scales=None, iou_threshold=0.5):
    """
    Create an anchor-based encoder function that handles metadata inputs.
    
    Args:
        num_classes: Number of object classes
        grid_size: (height, width) of output grid
        num_anchors: Number of anchors per grid cell
        anchor_scales: List of (width, height) anchor scales. If None, uses defaults.
        iou_threshold: Minimum IoU to match a box to an anchor
    
    Returns:
        Encoder function that takes ((images, metadata), targets) and returns 
        ((images, metadata), anchor_targets)
    """
    encoder = make_anchor_encoder(num_classes, grid_size, num_anchors, 
                                   anchor_scales, iou_threshold)
    
    def _tf_fn_with_meta(inputs, targets):
        """
        Encode targets to anchor format, preserving metadata.
        
        Args:
            inputs: Tuple of (images, metadata) or just images
            targets: Dict with "bboxes" and "labels"
        
        Returns:
            (inputs, anchor_targets) where inputs format is preserved
        """
        if isinstance(inputs, tuple):
            images, metadata = inputs
            images_encoded, anchor_targets = encoder(images, targets)
            return (images_encoded, metadata), anchor_targets
        else:
            images = inputs
            images_encoded, anchor_targets = encoder(images, targets)
            return images_encoded, anchor_targets
    
    return _tf_fn_with_meta


if __name__ == "__main__":
    # Test encoder
    print("Testing anchor encoder...")
    
    grid_size = (7, 7)
    num_classes = 10
    num_anchors = 3
    
    encoder = make_anchor_encoder(num_classes, grid_size, num_anchors)
    
    # Create dummy data
    batch_size = 2
    image_size = (224, 224)
    max_boxes = 5
    
    images = np.random.rand(batch_size, image_size[0], image_size[1], 3).astype(np.float32)
    bboxes = np.array([
        [[0.2, 0.2, 0.4, 0.4], [0.6, 0.6, 0.8, 0.8], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0.1, 0.1, 0.3, 0.3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ], dtype=np.float32)
    labels = np.array([
        [0, 1, 0, 0, 0],
        [2, 0, 0, 0, 0],
    ], dtype=np.int32)
    
    targets = {"bboxes": bboxes, "labels": labels}
    
    # Note: This is a TensorFlow function, so we'd need to test it in a TF context
    print(f"Encoder created for grid_size={grid_size}, num_classes={num_classes}, num_anchors={num_anchors}")
    print(f"Expected output shape: [B, {grid_size[0]}, {grid_size[1]}, {num_anchors * (1 + 4 + num_classes)}]")

