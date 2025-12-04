"""
Build two-stage detector without metadata (TSNM).
Uses two-stage transfer learning (frozen then unfrozen backbone).
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers

from ..utils.load_pretrained_detector import load_backbone


def build_detector(
    image_size=(320, 320),
    num_classes=1,
    backbone_type="ssd_mobilenet_v2",
    freeze_backbone=True,
):
    """
    Build an SSD-style detection model without metadata.
    Uses two-stage transfer learning.
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
    
    # Preprocess image
    if backbone_type.startswith("mobilenet") or backbone_type == "ssd_mobilenet_v2":
        x_img = keras.applications.mobilenet_v2.preprocess_input(image_input)
    elif backbone_type == "resnet50":
        x_img = keras.applications.resnet50.preprocess_input(image_input)
    elif backbone_type == "efficientnet_b0":
        x_img = keras.applications.efficientnet.preprocess_input(image_input)
    else:
        x_img = image_input / 255.0
    
    # Extract features from backbone
    feature_maps = backbone(x_img, training=not freeze_backbone)
    
    # Use the higher resolution feature map
    if isinstance(feature_maps, (list, tuple)):
        x_img = feature_maps[-1]
    else:
        x_img = feature_maps
    
    # Detection head - SSD-style (no metadata)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x_img)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    
    # Output layers
    obj_output = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="objectness")(x)
    bbox_output = layers.Conv2D(4, 1, padding="same", activation="sigmoid", name="bbox")(x)
    cls_output = layers.Conv2D(num_classes, 1, padding="same", activation="sigmoid", name="classes")(x)
    
    # Concatenate all outputs
    outputs = layers.Concatenate(axis=-1, name="detections")([obj_output, bbox_output, cls_output])
    
    model = keras.Model(
        inputs=image_input,
        outputs=outputs,
        name="tsnm_detector",
    )
    
    return model

