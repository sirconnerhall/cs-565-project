"""
YOLO-style CSPDarkNet backbone implementation.

Implements CSPDarkNet architecture (YOLOv5 style) for object detection.
Uses ImageNet pre-trained weights where possible.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


def conv_bn_act(x, filters, kernel_size, strides=1, activation='swish', name=None):
    """
    Convolution + BatchNorm + Activation block.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Kernel size (int or tuple)
        strides: Stride (default 1)
        activation: Activation function (default 'swish')
        name: Layer name prefix
    
    Returns:
        Output tensor
    """
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding='same',
        use_bias=False, name=f'{name}_conv' if name else None
    )(x)
    x = layers.BatchNormalization(name=f'{name}_bn' if name else None)(x)
    if activation == 'swish':
        x = layers.Activation(tf.nn.swish, name=f'{name}_act' if name else None)(x)
    elif activation:
        x = layers.Activation(activation, name=f'{name}_act' if name else None)(x)
    return x


def bottleneck(x, filters, shortcut=True, name=None):
    """
    Bottleneck block (residual connection).
    
    Args:
        x: Input tensor
        filters: Number of filters
        shortcut: Whether to use residual connection
        name: Layer name prefix
    
    Returns:
        Output tensor
    """
    y = conv_bn_act(x, filters, 1, name=f'{name}_conv1' if name else None)
    y = conv_bn_act(y, filters, 3, name=f'{name}_conv2' if name else None)
    if shortcut:
        y = layers.Add(name=f'{name}_add' if name else None)([x, y])
    return y


def csp_block(x, filters, num_blocks, shortcut=True, name=None):
    """
    CSP (Cross Stage Partial) block.
    
    Args:
        x: Input tensor
        filters: Number of filters
        num_blocks: Number of bottleneck blocks
        shortcut: Whether to use residual connections
        name: Layer name prefix
    
    Returns:
        Output tensor
    """
    # Split channels
    x1 = conv_bn_act(x, filters // 2, 1, name=f'{name}_split_conv' if name else None)
    x2 = conv_bn_act(x, filters // 2, 1, name=f'{name}_split_conv2' if name else None)
    
    # Apply bottlenecks to x2
    for i in range(num_blocks):
        x2 = bottleneck(x2, filters // 2, shortcut=shortcut, 
                       name=f'{name}_bottleneck_{i}' if name else None)
    
    # Concatenate
    y = layers.Concatenate(name=f'{name}_concat' if name else None)([x1, x2])
    y = conv_bn_act(y, filters, 1, name=f'{name}_out_conv' if name else None)
    return y


def build_cspdarknet_backbone(input_shape=(224, 224, 3), weights='imagenet'):
    """
    Build CSPDarkNet backbone (YOLOv5 style).
    
    Architecture:
    - Input: [H, W, 3]
    - Stem: 3x3 conv, stride 2
    - CSP blocks with downsampling
    - Output: Feature map at 32x downsampling
    
    Args:
        input_shape: Input image shape (height, width, channels)
        weights: 'imagenet' for ImageNet weights, None for random initialization
    
    Returns:
        Keras model with CSPDarkNet backbone
    """
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Stem
    x = conv_bn_act(inputs, 32, 3, strides=2, name='stem')  # /2
    
    # Stage 1
    x = conv_bn_act(x, 64, 3, strides=2, name='stage1_conv')  # /4
    x = csp_block(x, 64, num_blocks=1, name='stage1_csp')
    
    # Stage 2
    x = conv_bn_act(x, 128, 3, strides=2, name='stage2_conv')  # /8
    x = csp_block(x, 128, num_blocks=2, name='stage2_csp')
    
    # Stage 3
    x = conv_bn_act(x, 256, 3, strides=2, name='stage3_conv')  # /16
    x = csp_block(x, 256, num_blocks=3, name='stage3_csp')
    
    # Stage 4
    x = conv_bn_act(x, 512, 3, strides=2, name='stage4_conv')  # /32
    x = csp_block(x, 512, num_blocks=3, name='stage4_csp')
    
    model = keras.Model(inputs, x, name='cspdarknet_backbone')
    
    # Note: CSPDarkNet doesn't have ImageNet weights in Keras
    # Consider using EfficientNet-B0 or ResNet50 which have ImageNet weights
    if weights == 'imagenet':
        print("[CSPDarkNet] Warning: ImageNet pre-trained weights not available for CSPDarkNet.")
        print("[CSPDarkNet] Using random initialization.")
        print("[CSPDarkNet] Recommendation: Use 'efficientnet_b0' or 'resnet50' for ImageNet weights.")
    
    return model


def build_efficientnet_backbone(input_shape=(224, 224, 3), weights='imagenet'):
    """
    Build EfficientNet-B0 backbone with ImageNet pretrained weights.
    
    This is a good alternative to CSPDarkNet that has ImageNet weights available.
    EfficientNet-B0 provides excellent feature extraction for object detection.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        weights: 'imagenet' for ImageNet weights, None for random initialization
    
    Returns:
        Keras model with EfficientNet-B0 backbone
    """
    backbone = keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights=weights if weights == 'imagenet' else None,
    )
    
    # Get feature map at 32x downsampling (similar to CSPDarkNet)
    # EfficientNet-B0 downsamples by 32x at the final block
    model = keras.Model(inputs=backbone.input, outputs=backbone.output, name='efficientnet_b0_backbone')
    
    if weights == 'imagenet':
        print("[EfficientNet-B0] Loaded ImageNet pretrained weights successfully.")
    
    return model


def preprocess_input_cspdarknet(x):
    """
    Preprocess input for CSPDarkNet.
    
    Args:
        x: Input tensor in [0, 255] range
    
    Returns:
        Preprocessed tensor (normalized to [-1, 1])
    """
    # Normalize to [0, 1] then to [-1, 1] (similar to ImageNet preprocessing)
    # Input is expected to be in [0, 255] range, so always normalize
    # Use Lambda layer to wrap the preprocessing for Keras functional API
    from keras import layers
    
    def _preprocess(x_tensor):
        x_tensor = tf.cast(x_tensor, tf.float32)
        # Normalize from [0, 255] to [0, 1]
        x_tensor = x_tensor / 255.0
        # Normalize to [-1, 1]
        x_tensor = (x_tensor - 0.5) / 0.5
        return x_tensor
    
    # Return Lambda layer that can be used in functional API
    return layers.Lambda(_preprocess, name="preprocess_cspdarknet")(x)


def preprocess_input_efficientnet(x):
    """
    Preprocess input for EfficientNet (uses standard EfficientNet preprocessing).
    
    Args:
        x: Input tensor in [0, 255] range
    
    Returns:
        Preprocessed tensor
    """
    return keras.applications.efficientnet.preprocess_input(x)


if __name__ == "__main__":
    # Test backbone
    print("Testing CSPDarkNet backbone...")
    model = build_cspdarknet_backbone(input_shape=(224, 224, 3))
    model.summary()
    
    # Test forward pass
    test_input = tf.random.normal((1, 224, 224, 3))
    output = model(test_input, training=False)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Downsampling factor: {224 / output.shape[1]:.1f}x")

