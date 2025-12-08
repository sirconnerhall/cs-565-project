"""
FiLM (Feature-wise Linear Modulation) Layer for metadata conditioning.

This layer applies FiLM conditioning to image features using metadata.
Replaces Lambda layers to enable proper Keras model serialization.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


class FiLMLayer(keras.layers.Layer):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Applies scale and shift conditioning to image features based on metadata.
    This is a proper Keras layer (not Lambda) to enable model serialization.
    
    Args:
        c_feat: Number of channels in the feature map (int)
        name: Layer name
    """
    
    def __init__(self, c_feat, name="film_conditioning", **kwargs):
        super().__init__(name=name, **kwargs)
        # Store as Python int for JSON serialization
        self.c_feat = int(c_feat)
    
    def call(self, inputs):
        """
        Apply FiLM conditioning to image features.
        
        Args:
            inputs: List of two tensors [x_img, x_meta]
                - x_img: Image features [B, H, W, C]
                - x_meta: Metadata features [B, C * 2]
        
        Returns:
            Conditioned image features [B, H, W, C]
        """
        x_img_tensor, x_meta_tensor = inputs
        
        # Get dynamic spatial dimensions
        shape = tf.shape(x_img_tensor)
        h_feat = shape[1]
        w_feat = shape[2]
        
        # Reshape metadata to [B, 1, 1, C * 2]
        x_meta_reshaped = tf.reshape(x_meta_tensor, [-1, 1, 1, self.c_feat * 2])
        
        # Tile to match spatial dimensions [B, H, W, C * 2]
        x_meta_tiled = tf.tile(x_meta_reshaped, [1, h_feat, w_feat, 1])
        
        # Split into scale and shift
        scale = x_meta_tiled[..., :self.c_feat]  # [B, H, W, C]
        shift = x_meta_tiled[..., self.c_feat:]  # [B, H, W, C]
        
        # Apply FiLM conditioning: features * (1 + scale) + shift
        return x_img_tensor * (1.0 + scale) + shift
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "c_feat": int(self.c_feat),  # Ensure it's a Python int
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)

