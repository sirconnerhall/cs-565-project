"""
Build single-stage detector with metadata (SSM).
Backbone stays frozen throughout training.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_detector(
    image_size=(224, 224),
    num_classes=1,
    metadata_dim=5,
    backbone_type="mobilenet_v2",
):
    """
    Build a single-scale detection model with metadata.
    Backbone is frozen (single-stage transfer learning).
    """
    h, w = image_size
    input_shape = (h, w, 3)
    
    # Image input
    image_input = keras.Input(shape=input_shape, name="image")
    
    # Load backbone
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
        name="ssm_detector",
    )
    
    return model

