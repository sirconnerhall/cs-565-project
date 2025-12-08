"""
Debug utility to find which component is storing tensor values in model config.
"""

import json
import tensorflow as tf
from tensorflow import keras


def is_tensor_like(obj):
    """Check if an object is a TensorFlow tensor or tensor-like."""
    # In TF2, EagerTensor is just tf.Tensor
    tensor_types = (tf.Tensor, tf.Variable)
    # Also check for string representation to catch EagerTensor
    if hasattr(obj, 'numpy'):
        return True
    if isinstance(obj, tensor_types):
        return True
    # Check class name as fallback
    class_name = type(obj).__name__
    if 'Tensor' in class_name or 'EagerTensor' in class_name:
        return True
    return False


def find_tensor_in_dict(d, path="", tensors_found=None):
    """
    Recursively search a dictionary for tensor values.
    
    Returns list of paths where tensors are found.
    """
    if tensors_found is None:
        tensors_found = []
    
    if isinstance(d, dict):
        for key, value in d.items():
            current_path = f"{path}.{key}" if path else key
            if is_tensor_like(value):
                try:
                    value_str = str(value.numpy()) if hasattr(value, 'numpy') else str(value)
                    tensors_found.append((current_path, type(value).__name__, value_str[:100]))
                except:
                    tensors_found.append((current_path, type(value).__name__, str(value)[:100]))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    item_path = f"{current_path}[{i}]"
                    if is_tensor_like(item):
                        try:
                            value_str = str(item.numpy()) if hasattr(item, 'numpy') else str(item)
                            tensors_found.append((item_path, type(item).__name__, value_str[:100]))
                        except:
                            tensors_found.append((item_path, type(item).__name__, str(item)[:100]))
                    elif isinstance(item, (dict, list, tuple)):
                        find_tensor_in_dict(item, item_path, tensors_found)
            elif isinstance(value, (dict, list, tuple)):
                find_tensor_in_dict(value, current_path, tensors_found)
    elif isinstance(d, (list, tuple)):
        for i, item in enumerate(d):
            item_path = f"{path}[{i}]" if path else f"[{i}]"
            if is_tensor_like(item):
                try:
                    value_str = str(item.numpy()) if hasattr(item, 'numpy') else str(item)
                    tensors_found.append((item_path, type(item).__name__, value_str[:100]))
                except:
                    tensors_found.append((item_path, type(item).__name__, str(item)[:100]))
            elif isinstance(item, (dict, list, tuple)):
                find_tensor_in_dict(item, item_path, tensors_found)
    
    return tensors_found


def debug_model_config(model):
    """
    Debug a model's config to find tensor values.
    
    Args:
        model: Keras model to debug
    
    Returns:
        List of (path, type, value) tuples where tensors are found
    """
    print("=" * 60)
    print("Debugging Model Config for Tensor Values")
    print("=" * 60)
    
    try:
        config = model.get_config()
        print(f"Model config type: {type(config)}")
        print(f"Model config keys: {list(config.keys()) if isinstance(config, dict) else 'N/A'}")
        
        tensors = find_tensor_in_dict(config)
        
        if tensors:
            print(f"\nFound {len(tensors)} tensor(s) in model config:")
            for path, tensor_type, value in tensors:
                print(f"  - {path}: {tensor_type} = {value}")
        else:
            print("\nNo tensors found in model config (good!)")
        
        # Also check each layer's config
        print("\n" + "=" * 60)
        print("Checking individual layer configs:")
        print("=" * 60)
        
        layer_tensors = []
        for i, layer in enumerate(model.layers):
            try:
                layer_config = layer.get_config()
                # Try to serialize each layer's config individually
                try:
                    json.dumps(layer_config, default=str)
                except (TypeError, ValueError) as ser_err:
                    print(f"\n⚠ Layer {i}: {layer.name} ({type(layer).__name__}) - CANNOT BE SERIALIZED")
                    print(f"   Error: {ser_err}")
                    layer_tensors_found = find_tensor_in_dict(layer_config, f"layers[{i}].{layer.name}")
                    if layer_tensors_found:
                        layer_tensors.extend(layer_tensors_found)
                        for path, tensor_type, value in layer_tensors_found:
                            print(f"  - {path}: {tensor_type} = {value}")
                    else:
                        # If we can't find it with our finder, try deeper inspection
                        print(f"   (Tensor found but not detected by finder - may be nested)")
                        # Try to find it by attempting to access each key
                        for key in layer_config.keys():
                            try:
                                json.dumps(layer_config[key], default=str)
                            except:
                                print(f"   Problematic key: {key} (value type: {type(layer_config[key])})")
                                if hasattr(layer_config[key], 'numpy'):
                                    try:
                                        print(f"   Value: {layer_config[key].numpy()}")
                                    except:
                                        print(f"   Value: {layer_config[key]}")
                else:
                    # Layer can be serialized, but check for tensors anyway
                    layer_tensors_found = find_tensor_in_dict(layer_config, f"layers[{i}].{layer.name}")
                    if layer_tensors_found:
                        layer_tensors.extend(layer_tensors_found)
                        print(f"\nLayer {i}: {layer.name} ({type(layer).__name__}) - has tensors but serializes")
                        for path, tensor_type, value in layer_tensors_found:
                            print(f"  - {path}: {tensor_type} = {value}")
            except Exception as e:
                print(f"\nLayer {i}: {layer.name} - Error getting config: {e}")
        
        if not layer_tensors:
            print("\nNo tensors found in any layer configs (but serialization still fails - checking deeper...)")
        
        return tensors + layer_tensors
        
    except Exception as e:
        print(f"\nError debugging model config: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_model_serialization(model, model_name="test_model"):
    """
    Test if a model can be serialized to JSON.
    
    Args:
        model: Keras model to test
        model_name: Name for the model
    
    Returns:
        True if serialization succeeds, False otherwise
    """
    print("\n" + "=" * 60)
    print(f"Testing JSON Serialization for {model_name}")
    print("=" * 60)
    
    try:
        config = model.get_config()
        json_str = json.dumps(config, indent=2, default=str)  # Use default=str to catch non-serializable
        print("✓ Model config successfully serialized to JSON")
        return True
    except (TypeError, ValueError) as e:
        print(f"✗ Model config serialization failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        
        # Try to find the problematic value by attempting serialization with custom handler
        try:
            config = model.get_config()
            
            # Custom JSON encoder that identifies problematic values
            class TensorEncoder(json.JSONEncoder):
                def default(self, obj):
                    if is_tensor_like(obj):
                        return f"<TENSOR: {type(obj).__name__}>"
                    return super().default(obj)
            
            # Try to serialize and catch where it fails
            def find_problematic_value(obj, path=""):
                try:
                    json.dumps(obj, cls=TensorEncoder)
                except TypeError as te:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            find_problematic_value(v, f"{path}.{k}" if path else k)
                    elif isinstance(obj, (list, tuple)):
                        for i, item in enumerate(obj):
                            find_problematic_value(item, f"{path}[{i}]" if path else f"[{i}]")
                    else:
                        if is_tensor_like(obj):
                            print(f"    - Found tensor at: {path} (type: {type(obj).__name__})")
            
            print(f"\n  Searching for problematic values...")
            find_problematic_value(config)
            
            # Also use our tensor finder
            tensors = find_tensor_in_dict(config)
            if tensors:
                print(f"\n  Found {len(tensors)} tensor(s) causing the issue:")
                for path, tensor_type, value in tensors[:10]:  # Show first 10
                    print(f"    - {path}: {tensor_type} = {value}")
        except Exception as debug_e:
            print(f"  Could not debug further: {debug_e}")
        
        return False

