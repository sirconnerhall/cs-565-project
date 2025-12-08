"""
Load pre-trained object detection models for transfer learning.

Uses Keras applications to build detection backbones with ImageNet pre-trained weights.
This approach is simpler and doesn't require TensorFlow Hub.
"""

import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import layers

from .yolo_backbone import build_cspdarknet_backbone, preprocess_input_cspdarknet


def load_backbone(model_type="ssd_mobilenet_v2", input_shape=(640, 640, 3)):
    """
    Load a pre-trained backbone for object detection.
    
    Args:
        model_type: One of "ssd_mobilenet_v2", "mobilenet_v2", "resnet50", "efficientnet_b0", "cspdarknet"
        input_shape: Input image shape (height, width, channels)
    
    Returns:
        Keras model (backbone only, no detection head)
    """
    print(f"[Model] Loading {model_type} backbone...")
    
    if model_type == "cspdarknet" or model_type == "yolo_backbone":
        # CSPDarkNet backbone (YOLO-style) - no ImageNet weights available
        from .yolo_backbone import build_cspdarknet_backbone
        backbone = build_cspdarknet_backbone(input_shape=input_shape, weights='imagenet')
        # Return single feature map (not multi-scale for now)
        model = keras.Model(inputs=backbone.input, outputs=backbone.output, name="cspdarknet_backbone")
        print(f"[Model] CSPDarkNet backbone loaded (no ImageNet weights, random initialization)")
        print(f"[Model] Recommendation: Use 'efficientnet_b0' for ImageNet pretrained weights")
        
    elif model_type == "efficientnet_b0" or model_type == "efficientnet":
        # EfficientNet-B0 with ImageNet weights (recommended alternative)
        from .yolo_backbone import build_efficientnet_backbone
        backbone = build_efficientnet_backbone(input_shape=input_shape, weights='imagenet')
        model = keras.Model(inputs=backbone.input, outputs=backbone.output, name="efficientnet_b0_backbone")
        
    elif model_type == "ssd_mobilenet_v2" or model_type == "mobilenet_v2":
        backbone = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        # Get feature maps at different scales for multi-scale detection
        layer_names = [
            "block_5_expand",   # Lower resolution (for larger objects)
            "block_12_expand",  # Higher resolution (for smaller objects)
        ]
        outputs = [backbone.get_layer(name).output for name in layer_names]
        model = keras.Model(inputs=backbone.input, outputs=outputs, name="mobilenet_v2_backbone")
        
    elif model_type == "resnet50":
        backbone = keras.applications.ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        # Get feature maps from different stages
        layer_names = [
            "conv3_block4_out",  # Stage 3
            "conv4_block6_out",   # Stage 4
        ]
        outputs = [backbone.get_layer(name).output for name in layer_names]
        model = keras.Model(inputs=backbone.input, outputs=outputs, name="resnet50_backbone")
        
    elif model_type == "efficientnet_b0":
        backbone = keras.applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
        # Get feature maps from different blocks
        layer_names = [
            "block3a_expand_activation",  # Mid-level features
            "block6a_expand_activation",  # High-level features
        ]
        outputs = [backbone.get_layer(name).output for name in layer_names]
        model = keras.Model(inputs=backbone.input, outputs=outputs, name="efficientnet_b0_backbone")
        
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: cspdarknet, efficientnet_b0, ssd_mobilenet_v2, mobilenet_v2, resnet50"
        )
    
    print(f"[Model] Successfully loaded {model_type} backbone")
    return model


def get_model_info(model_type):
    """
    Get information about a model type.
    
    Args:
        model_type: Model type identifier
    
    Returns:
        Dict with model information (input_size, num_classes, etc.)
    """
    info = {
        "cspdarknet": {
            "input_size": (224, 224),
            "description": "CSPDarkNet - YOLO-style backbone (no ImageNet weights)",
        },
        "efficientnet_b0": {
            "input_size": (224, 224),
            "description": "EfficientNet-B0 - ImageNet pretrained (recommended)",
        },
        "ssd_mobilenet_v2": {
            "input_size": (320, 320),
            "description": "SSD MobileNet V2 - Fast, lightweight",
        },
        "mobilenet_v2": {
            "input_size": (224, 224),
            "description": "MobileNet V2 - Standard",
        },
        "resnet50": {
            "input_size": (224, 224),
            "description": "ResNet50 - Higher accuracy, slower",
        },
        "efficientnet_b0": {
            "input_size": (224, 224),
            "description": "EfficientNet-B0 - Good balance",
        },
    }
    
    return info.get(model_type, {})


if __name__ == "__main__":
    # Test model loading
    print("Testing model loading...")
    model = load_backbone("ssd_mobilenet_v2", input_shape=(320, 320, 3))
    print(f"Model type: {type(model)}")
    print(f"Model summary:")
    model.summary()
