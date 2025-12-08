"""
Build YOLO-style detection model with metadata integration.

Uses CSPDarkNet backbone with anchor-based detection and metadata integration.
- Detection head for bounding boxes and classes with anchors
- Metadata branch (location, time, brightness)
- FiLM conditioning to fuse metadata with image features
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


def build_ssd_detector_with_metadata(
    image_size=(320, 320),
    num_classes=1,
    metadata_dim=8,  # Updated: 8 features with cyclical encoding (location, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, brightness)
    backbone_type="cspdarknet",
    freeze_backbone=True,
    num_anchors=3,
):
    """
    Build an anchor-based detection model with metadata integration.
    
    Architecture:
    1. Pre-trained CSPDarkNet backbone - extracts image features
    2. Metadata branch - processes location, time (with cyclical encoding), brightness
    3. FiLM conditioning - injects metadata into image features
    4. Detection head - predicts bounding boxes and classes with anchors
    
    Args:
        image_size: (height, width) of input images
        num_classes: Number of object classes (excluding background)
        metadata_dim: Dimension of metadata vector (default 8: location, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, brightness)
        backbone_type: Type of backbone (default "cspdarknet")
        freeze_backbone: Whether to freeze backbone weights initially
        num_anchors: Number of anchors per grid cell (default 3)
    
    Returns:
        Keras model with inputs [image, metadata] and output anchor-based detection predictions
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
    
    # Metadata input
    metadata_input = keras.Input(shape=(metadata_dim,), name="metadata")
    
    # Metadata branch
    x_meta = layers.Dense(64, activation="relu")(metadata_input)
    x_meta = layers.Dense(32, activation="relu")(x_meta)
    x_meta = layers.Dense(16, activation="relu")(x_meta)  # [B, 16]
    
    # Expand metadata to spatial dimensions for FiLM conditioning
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
    x_meta_expanded = layers.Dense(c_feat * 2)(x_meta)  # [B, C' * 2]
    
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
        name="ssd_detector_with_metadata",
    )
    
    return model


def build_simple_detector_with_metadata(
    image_size=(224, 224),
    num_classes=1,
    metadata_dim=8,  # Updated: 8 features with cyclical encoding
    backbone_type="mobilenet_v2",
    freeze_backbone=True,
):
    """
    Build a simpler single-scale detection model with metadata.
    
    Similar to build_ssd_detector_with_metadata but uses single feature map
    instead of multi-scale features. Faster but potentially less accurate.
    """
    h, w = image_size
    input_shape = (h, w, 3)
    
    # Image input
    image_input = keras.Input(shape=input_shape, name="image")
    
    # Load backbone (single output, not multi-scale)
    if backbone_type == "mobilenet_v2" or backbone_type == "ssd_mobilenet_v2":
        backbone = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        preprocess_fn = keras.applications.mobilenet_v2.preprocess_input
    elif backbone_type == "resnet50":
        backbone = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        preprocess_fn = keras.applications.resnet50.preprocess_input
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}")
    
    if freeze_backbone:
        backbone.trainable = False
    
    # Preprocess and extract features
    x_img = preprocess_fn(image_input)
    x_img = backbone(x_img, training=not freeze_backbone)
    
    # Metadata input and processing (same as SSD version)
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
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x_img)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    
    # Outputs
    obj_output = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="objectness")(x)
    bbox_output = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="bbox")(x)
    cls_output = layers.Conv2D(num_classes, 1, padding="same", activation="sigmoid", name="classes")(x)
    
    outputs = layers.Concatenate(axis=-1, name="detections")([obj_output, bbox_output, cls_output])
    
    model = keras.Model(
        inputs=[image_input, metadata_input],
        outputs=outputs,
        name="simple_detector_with_metadata",
    )
    
    return model


if __name__ == "__main__":
    # Test model building
    print("Testing detector model building...")
    model = build_ssd_detector_with_metadata(
        image_size=(320, 320),
        num_classes=10,
        metadata_dim=8,  # Updated: 8 features with cyclical encoding
        backbone_type="ssd_mobilenet_v2",
        freeze_backbone=True,
    )
    print("\nModel summary:")
    model.summary()
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_image = tf.random.normal((1, 320, 320, 3))
    test_metadata = tf.random.normal((1, 5))
    output = model([test_image, test_metadata])
    print(f"Output shape: {output.shape}")

