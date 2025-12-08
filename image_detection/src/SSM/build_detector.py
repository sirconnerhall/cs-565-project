"""
Build single-stage detector with metadata (SSM).
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
from ..utils.film_layer import FiLMLayer


def build_detector(
    image_size=(224, 224),
    num_classes=1,
    metadata_dim=8,  # Updated: 8 features with cyclical encoding (location, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, brightness)
    backbone_type="cspdarknet",
    num_anchors=3,
):
    """
    Build a single-scale detection model with metadata and anchor-based detection.
    Backbone is frozen (single-stage transfer learning).
    
    Args:
        image_size: (height, width) of input images
        num_classes: Number of object classes
        metadata_dim: Dimension of metadata vector (default 8)
        backbone_type: Backbone type (default "cspdarknet")
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
    
    backbone.trainable = False  # Single stage: always frozen
    
    # Preprocess and extract features
    x_img = preprocess_fn(image_input)
    x_img = backbone(x_img, training=False)
    
    # Metadata input and processing
    metadata_input = keras.Input(shape=(metadata_dim,), name="metadata")
    x_meta = layers.Dense(64, activation="relu")(metadata_input)
    x_meta = layers.Dense(32, activation="relu")(x_meta)
    x_meta = layers.Dense(16, activation="relu")(x_meta)
    
    # FiLM conditioning
    # Get channel dimension from static shape (needed for Dense layer initialization)
    # Use backbone output shape directly to avoid any tensor storage
    # output_shape is a tuple like (None, H, W, C) - batch dim may be None
    backbone_output_shape = backbone.output_shape
    if backbone_output_shape is None:
        raise ValueError("Backbone output_shape is None - backbone may not be built")
    if len(backbone_output_shape) != 4:
        raise ValueError(f"Backbone must output 4D feature map [B, H, W, C], got shape: {backbone_output_shape}")
    # Get channel dimension (last element) - ensure it's not None and convert to int
    c_feat_raw = backbone_output_shape[-1]
    if c_feat_raw is None:
        raise ValueError(f"Backbone output channel dimension is None in shape: {backbone_output_shape}")
    c_feat = int(c_feat_raw)  # Get channel dimension as Python int
    
    # Expand metadata to match feature map channels
    x_meta_expanded = layers.Dense(c_feat * 2)(x_meta)
    
    # Apply FiLM conditioning using custom layer (enables proper model serialization)
    x_img = FiLMLayer(c_feat=c_feat, name="film_conditioning")([x_img, x_meta_expanded])
    
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
        inputs=[image_input, metadata_input],
        outputs=outputs,
        name="ssm_detector",
    )
    
    return model

