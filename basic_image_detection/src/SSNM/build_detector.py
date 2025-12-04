"""
Build single-stage detector without metadata.
Backbone stays frozen throughout training.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


def build_detector(image_size, num_classes):
    """
    Simple grid-based detector on top of MobileNetV2.
    Output shape: [B, S, S, 5 + C] with sigmoid activation.
    Backbone is frozen (single-stage transfer learning).
    """
    h, w = image_size
    inputs = keras.Input(shape=(h, w, 3), name="image")

    base_model = keras.applications.MobileNetV2(
        input_shape=(h, w, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Single stage: always frozen

    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)

    # A small conv head
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)

    # 5 + num_classes channels: [obj, cx, cy, w, h, class_probs...]
    outputs = layers.Conv2D(
        5 + num_classes,
        1,
        padding="same",
        activation="sigmoid",  # all outputs in [0, 1]
        name="grid_output",
    )(x)

    model = keras.Model(inputs, outputs, name="ssnm_detector")
    return model

