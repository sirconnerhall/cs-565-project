"""
Build single-stage detector without metadata.
Backbone stays frozen throughout training.
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


def build_detector(image_size, num_classes, num_anchors=3, backbone_type="cspdarknet"):
    """
    Anchor-based detector on top of CSPDarkNet (YOLO-style).
    Output shape: [B, H, W, num_anchors * (1 + 4 + num_classes)] with sigmoid activation.
    Backbone is frozen (single-stage transfer learning).
    
    Args:
        image_size: (height, width) of input images
        num_classes: Number of object classes
        num_anchors: Number of anchors per grid cell (default 3)
        backbone_type: Backbone type (default "cspdarknet")
    
    Returns:
        Keras model
    """
    h, w = image_size
    inputs = keras.Input(shape=(h, w, 3), name="image")

    # Build backbone (use EfficientNet-B0 if available, otherwise CSPDarkNet)
    if backbone_type == "efficientnet_b0" or backbone_type == "efficientnet":
        backbone = build_efficientnet_backbone(
            input_shape=(h, w, 3),
            weights='imagenet'
        )
        preprocess_fn = preprocess_input_efficientnet
    else:
        # Default to CSPDarkNet (no ImageNet weights)
        backbone = build_cspdarknet_backbone(
            input_shape=(h, w, 3),
            weights='imagenet'
        )
        preprocess_fn = preprocess_input_cspdarknet
    
    backbone.trainable = False  # Single stage: always frozen

    # Preprocess and extract features
    x = preprocess_fn(inputs)
    x = backbone(x, training=False)

    # Detection head
    x = layers.Conv2D(256, 3, padding="same", activation="swish")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="swish")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="swish")(x)

    # Output: num_anchors * (objectness + 4 offsets + num_classes) per grid cell
    depth_per_anchor = 1 + 4 + num_classes
    total_depth = num_anchors * depth_per_anchor
    
    outputs = layers.Conv2D(
        total_depth,
        1,
        padding="same",
        activation="sigmoid",  # all outputs in [0, 1]
        name="anchor_output",
    )(x)

    model = keras.Model(inputs, outputs, name="ssnm_detector")
    return model

