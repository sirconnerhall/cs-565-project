"""
Build YOLO-style detection model with metadata integration.

Uses pre-trained backbone from load_pretrained_detector and adds:
- Detection head for bounding boxes and classes
- Metadata branch (location, time, brightness)
- FiLM conditioning to fuse metadata with image features
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

from load_pretrained_detector import load_backbone


def build_ssd_detector_with_metadata(
    image_size=(320, 320),
    num_classes=1,
    metadata_dim=5,
    backbone_type="ssd_mobilenet_v2",
    freeze_backbone=True,
):
    """
    Build an SSD-style detection model with metadata integration.
    
    Architecture:
    1. Pre-trained backbone (MobileNetV2, ResNet50, etc.) - extracts image features
    2. Metadata branch - processes location, time, brightness
    3. FiLM conditioning - injects metadata into image features
    4. Detection head - predicts bounding boxes and classes
    
    Args:
        image_size: (height, width) of input images
        num_classes: Number of object classes (excluding background)
        metadata_dim: Dimension of metadata vector (default 5: location, hour, day, month, brightness)
        backbone_type: Type of backbone ("ssd_mobilenet_v2", "resnet50", etc.)
        freeze_backbone: Whether to freeze backbone weights initially
    
    Returns:
        Keras model with inputs [image, metadata] and output detection predictions
    """
    h, w = image_size
    input_shape = (h, w, 3)
    
    # Image input
    image_input = keras.Input(shape=input_shape, name="image")
    
    # Load pre-trained backbone
    backbone = load_backbone(backbone_type, input_shape=input_shape)
    
    if freeze_backbone:
        backbone.trainable = False
        print("[Model] Backbone frozen")
    else:
        backbone.trainable = True
        print("[Model] Backbone trainable")
    
    # Preprocess image (normalize for backbone)
    if backbone_type.startswith("mobilenet") or backbone_type == "ssd_mobilenet_v2":
        x_img = keras.applications.mobilenet_v2.preprocess_input(image_input)
    elif backbone_type == "resnet50":
        x_img = keras.applications.resnet50.preprocess_input(image_input)
    elif backbone_type == "efficientnet_b0":
        x_img = keras.applications.efficientnet.preprocess_input(image_input)
    else:
        x_img = image_input / 255.0  # Simple normalization
    
    # Extract multi-scale features from backbone
    # Backbone returns list of feature maps at different scales
    feature_maps = backbone(x_img, training=not freeze_backbone)
    
    # Use the higher resolution feature map for detection
    # (last one in the list is typically higher resolution)
    if isinstance(feature_maps, (list, tuple)):
        x_img = feature_maps[-1]  # Use highest resolution features
    else:
        x_img = feature_maps
    
    # Metadata input
    metadata_input = keras.Input(shape=(metadata_dim,), name="metadata")
    
    # Metadata branch
    x_meta = layers.Dense(64, activation="relu")(metadata_input)
    x_meta = layers.Dense(32, activation="relu")(x_meta)
    x_meta = layers.Dense(16, activation="relu")(x_meta)  # [B, 16]
    
    # Expand metadata to spatial dimensions for FiLM conditioning
    _, h_feat, w_feat, c_feat = x_img.shape
    x_meta_expanded = layers.Dense(c_feat * 2)(x_meta)  # [B, C' * 2]
    x_meta_expanded = tf.reshape(x_meta_expanded, [-1, 1, 1, c_feat * 2])  # [B, 1, 1, C' * 2]
    x_meta_expanded = tf.tile(x_meta_expanded, [1, h_feat, w_feat, 1])  # [B, H', W', C' * 2]
    
    # Split into scale and shift (FiLM conditioning)
    scale = x_meta_expanded[..., :c_feat]  # [B, H', W', C']
    shift = x_meta_expanded[..., c_feat:]  # [B, H', W', C']
    
    # Apply FiLM conditioning
    x_img = x_img * (1.0 + scale) + shift
    
    # Detection head - SSD-style
    # Use multiple convolutional layers to refine features
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x_img)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    
    # Output layers
    # For each spatial location, predict:
    # - Objectness (1 value): probability of object presence
    # - Bounding box (4 values): cx, cy, w, h (normalized)
    # - Class probabilities (num_classes values)
    
    # Objectness branch
    obj_output = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="objectness")(x)
    
    # Bounding box branch
    bbox_output = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="bbox")(x)
    
    # Class branch
    cls_output = layers.Conv2D(num_classes, 1, padding="same", activation="sigmoid", name="classes")(x)
    
    # Concatenate all outputs: [B, H, W, 1 + 4 + num_classes]
    outputs = layers.Concatenate(axis=-1, name="detections")([obj_output, bbox_output, cls_output])
    
    model = keras.Model(
        inputs=[image_input, metadata_input],
        outputs=outputs,
        name="ssd_detector_with_metadata",
    )
    
    return model


def build_simple_detector_with_metadata(
    image_size=(224, 224),
    num_classes=1,
    metadata_dim=5,
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
    _, h_feat, w_feat, c_feat = x_img.shape
    x_meta_expanded = layers.Dense(c_feat * 2)(x_meta)
    x_meta_expanded = tf.reshape(x_meta_expanded, [-1, 1, 1, c_feat * 2])
    x_meta_expanded = tf.tile(x_meta_expanded, [1, h_feat, w_feat, 1])
    
    scale = x_meta_expanded[..., :c_feat]
    shift = x_meta_expanded[..., c_feat:]
    x_img = x_img * (1.0 + scale) + shift
    
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
        metadata_dim=5,
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

