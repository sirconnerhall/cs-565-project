"""
Build two-stage detector without metadata (TSNM).
Uses two-stage transfer learning (frozen then unfrozen backbone).
Uses CSPDarkNet backbone with anchor-based detection.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

from ..utils.yolo_backbone import (
    build_cspdarknet_backbone,
    build_efficientnet_backbone,
    preprocess_input_cspdarknet,
    preprocess_input_efficientnet,
)


def build_detector(
    image_size=(320, 320),
    num_classes=1,
    backbone_type="cspdarknet",
    freeze_backbone=True,
    num_anchors=3,
):
    """
    Build an anchor-based detection model with CSPDarkNet backbone.
    Uses two-stage transfer learning.
    
    Args:
        image_size: (height, width) of input images
        num_classes: Number of object classes
        backbone_type: Type of backbone (default "cspdarknet")
        freeze_backbone: Whether to freeze backbone weights initially
        num_anchors: Number of anchors per grid cell (default 3)
    """
    h, w = image_size
    input_shape = (h, w, 3)
    
    # Image input
    image_input = keras.Input(shape=input_shape, name="image")
    
    # Build backbone (use EfficientNet-B0 if available, otherwise CSPDarkNet)
    if backbone_type == "efficientnet_b0" or backbone_type == "efficientnet":
        backbone = build_efficientnet_backbone(
            input_shape=input_shape,
            weights='imagenet'
        )
        preprocess_fn = preprocess_input_efficientnet
    else:
        # Default to CSPDarkNet (no ImageNet weights)
        backbone = build_cspdarknet_backbone(
            input_shape=input_shape,
            weights='imagenet'
        )
        preprocess_fn = preprocess_input_cspdarknet
    
    if freeze_backbone:
        backbone.trainable = False
        print("[Model] Backbone frozen")
    else:
        backbone.trainable = True
        print("[Model] Backbone trainable")
    
    # Preprocess and extract features
    x_img = preprocess_fn(image_input)
    x_img = backbone(x_img, training=not freeze_backbone)
    
    # Detection head
    x = layers.Conv2D(256, 3, padding="same", activation="swish")(x_img)
    x = layers.Conv2D(256, 3, padding="same", activation="swish")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="swish")(x)
    
    # Output: num_anchors * (objectness + 4 offsets + num_classes) per grid cell
    depth_per_anchor = 1 + 4 + num_classes
    total_depth = num_anchors * depth_per_anchor
    
    outputs = layers.Conv2D(
        total_depth,
        1,
        padding="same",
        activation="sigmoid",
        name="anchor_detections",
    )(x)
    
    model = keras.Model(
        inputs=image_input,
        outputs=outputs,
        name="tsnm_detector",
    )
    
    return model

