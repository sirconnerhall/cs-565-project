"""
Metadata encoding utilities with cyclical encoding for temporal features.

Implements cyclical encoding using sine-cosine functions for temporal features
to preserve periodic nature (e.g., hour 23:00 is close to hour 00:00).
"""

import numpy as np
import tensorflow as tf
from datetime import datetime


def cyclical_encode_hour(hour, max_hour=24):
    """
    Encode hour using cyclical encoding (sine-cosine).
    
    Args:
        hour: Hour value (0-23) or normalized (0-1)
        max_hour: Maximum hour value (default 24)
    
    Returns:
        tuple: (sin, cos) encoding
    """
    # If hour is already normalized [0, 1], convert back to [0, max_hour]
    if isinstance(hour, (int, float)) and hour <= 1.0:
        hour = hour * (max_hour - 1)
    
    # Ensure hour is in valid range
    hour = np.clip(hour, 0, max_hour - 1)
    
    sin_val = np.sin(2 * np.pi * hour / max_hour)
    cos_val = np.cos(2 * np.pi * hour / max_hour)
    
    return sin_val, cos_val


def cyclical_encode_day_of_week(day_of_week, max_day=7):
    """
    Encode day of week using cyclical encoding (sine-cosine).
    
    Args:
        day_of_week: Day value (0-6) or normalized (0-1)
        max_day: Maximum day value (default 7)
    
    Returns:
        tuple: (sin, cos) encoding
    """
    # If day_of_week is already normalized [0, 1], convert back to [0, max_day-1]
    if isinstance(day_of_week, (int, float)) and day_of_week <= 1.0:
        day_of_week = day_of_week * (max_day - 1)
    
    # Ensure day_of_week is in valid range
    day_of_week = np.clip(day_of_week, 0, max_day - 1)
    
    sin_val = np.sin(2 * np.pi * day_of_week / max_day)
    cos_val = np.cos(2 * np.pi * day_of_week / max_day)
    
    return sin_val, cos_val


def cyclical_encode_month(month, max_month=12):
    """
    Encode month using cyclical encoding (sine-cosine).
    
    Args:
        month: Month value (1-12) or normalized (0-1)
        max_month: Maximum month value (default 12)
    
    Returns:
        tuple: (sin, cos) encoding
    """
    # If month is already normalized [0, 1], convert back to [1, max_month]
    if isinstance(month, (int, float)) and month <= 1.0:
        month = 1 + month * (max_month - 1)
    
    # Ensure month is in valid range [1, max_month]
    month = np.clip(month, 1, max_month)
    
    sin_val = np.sin(2 * np.pi * month / max_month)
    cos_val = np.cos(2 * np.pi * month / max_month)
    
    return sin_val, cos_val


def encode_metadata_cyclical_numpy(location_id, hour, day_of_week, month, brightness):
    """
    Encode metadata with cyclical encoding for temporal features (NumPy version).
    
    Args:
        location_id: Location ID (normalized 0-1)
        hour: Hour value (0-23) or normalized (0-1)
        day_of_week: Day of week (0-6) or normalized (0-1)
        month: Month value (1-12) or normalized (0-1)
        brightness: Brightness value (0-1)
    
    Returns:
        numpy array of shape (8,) with cyclical encoding:
        [location_id, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, brightness]
    """
    hour_sin, hour_cos = cyclical_encode_hour(hour)
    day_sin, day_cos = cyclical_encode_day_of_week(day_of_week)
    month_sin, month_cos = cyclical_encode_month(month)
    
    return np.array([
        location_id,      # 1 feature
        hour_sin,         # 2 features (hour)
        hour_cos,
        day_sin,          # 2 features (day_of_week)
        day_cos,
        month_sin,        # 2 features (month)
        month_cos,
        brightness        # 1 feature
    ], dtype=np.float32)


def encode_metadata_cyclical_tf(location_id, hour, day_of_week, month, brightness):
    """
    Encode metadata with cyclical encoding for temporal features (TensorFlow version).
    
    Args:
        location_id: Location ID tensor (normalized 0-1) [B] or scalar
        hour: Hour value tensor (0-23) or normalized (0-1) [B] or scalar
        day_of_week: Day of week tensor (0-6) or normalized (0-1) [B] or scalar
        month: Month value tensor (1-12) or normalized (0-1) [B] or scalar
        brightness: Brightness tensor (0-1) [B] or scalar
    
    Returns:
        Tensor of shape [B, 8] or [8] with cyclical encoding:
        [location_id, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, brightness]
    """
    # Convert to float32
    location_id = tf.cast(location_id, tf.float32)
    hour = tf.cast(hour, tf.float32)
    day_of_week = tf.cast(day_of_week, tf.float32)
    month = tf.cast(month, tf.float32)
    brightness = tf.cast(brightness, tf.float32)
    
    # Assume inputs are in their natural ranges (not normalized)
    # Hour: clip to [0, 23]
    hour = tf.clip_by_value(hour, 0.0, 23.0)
    
    # Day of week: clip to [0, 6]
    day_of_week = tf.clip_by_value(day_of_week, 0.0, 6.0)
    
    # Month: clip to [1, 12]
    month = tf.clip_by_value(month, 1.0, 12.0)
    
    # Cyclical encoding
    hour_sin = tf.sin(2 * np.pi * hour / 24.0)
    hour_cos = tf.cos(2 * np.pi * hour / 24.0)
    
    day_sin = tf.sin(2 * np.pi * day_of_week / 7.0)
    day_cos = tf.cos(2 * np.pi * day_of_week / 7.0)
    
    month_sin = tf.sin(2 * np.pi * month / 12.0)
    month_cos = tf.cos(2 * np.pi * month / 12.0)
    
    # Stack into [B, 8] or [8]
    # Use axis=-1 to handle both batched and unbatched cases correctly
    metadata = tf.stack([
        location_id,
        hour_sin,
        hour_cos,
        day_sin,
        day_cos,
        month_sin,
        month_cos,
        brightness
    ], axis=-1)
    
    return metadata


def _parse_date_captured_py(date_str):
    """
    Parse date_captured string using Python datetime (for use with tf.py_function).
    
    Args:
        date_str: String with format "YYYY-MM-DD HH:MM:SS" (e.g., "2013-10-04 13:31:53")
    
    Returns:
        Tuple of (hour, day_of_week, month) as numpy int32:
        - hour: int32, range [0, 23]
        - day_of_week: int32, range [0, 6] (0=Monday, 6=Sunday)
        - month: int32, range [1, 12]
    """
    try:
        dt = datetime.strptime(date_str.decode('utf-8') if isinstance(date_str, bytes) else date_str, "%Y-%m-%d %H:%M:%S")
        hour = np.int32(dt.hour)
        day_of_week = np.int32(dt.weekday())  # 0=Monday, 6=Sunday
        month = np.int32(dt.month)
        return hour, day_of_week, month
    except (ValueError, AttributeError, TypeError):
        # Return default values if parsing fails
        return np.int32(12), np.int32(3), np.int32(6)


def parse_date_captured_tf(date_captured_str):
    """
    Parse date_captured string and extract temporal features (TensorFlow version).
    
    Args:
        date_captured_str: Tensor of strings with format "YYYY-MM-DD HH:MM:SS" (e.g., "2013-10-04 13:31:53")
                          Can be batched [B] or scalar []
    
    Returns:
        Tuple of (hour, day_of_week, month) tensors:
        - hour: [B] or scalar, int32, range [0, 23]
        - day_of_week: [B] or scalar, int32, range [0, 6] (0=Monday, 6=Sunday)
        - month: [B] or scalar, int32, range [1, 12]
        
        If parsing fails, returns default values (12, 3, 6).
    """
    # Use tf.py_function to call Python datetime parser
    # This is more reliable than trying to implement date parsing in pure TensorFlow
    def _parse_batch(date_strs):
        # Handle both batched and unbatched cases
        if date_strs.ndim == 0:
            date_strs = date_strs[np.newaxis]
        
        hours = []
        day_of_weeks = []
        months = []
        
        for date_str in date_strs:
            h, dow, m = _parse_date_captured_py(date_str)
            hours.append(h)
            day_of_weeks.append(dow)
            months.append(m)
        
        return np.array(hours, dtype=np.int32), np.array(day_of_weeks, dtype=np.int32), np.array(months, dtype=np.int32)
    
    # Use tf.py_function to call the Python parser
    hour, day_of_week, month = tf.py_function(
        _parse_batch,
        [date_captured_str],
        [tf.int32, tf.int32, tf.int32]
    )
    
    # Set shapes (handle both batched and unbatched)
    if date_captured_str.shape.ndims == 0:
        hour.set_shape([])
        day_of_week.set_shape([])
        month.set_shape([])
    else:
        hour.set_shape([None])
        day_of_week.set_shape([None])
        month.set_shape([None])
    
    # Clip values to valid ranges
    hour = tf.clip_by_value(hour, 0, 23)
    day_of_week = tf.clip_by_value(day_of_week, 0, 6)
    month = tf.clip_by_value(month, 1, 12)
    
    return hour, day_of_week, month


def parse_location_tf(location_str, max_location_id=200):
    """
    Parse location string and normalize to [0, 1] range.
    
    Args:
        location_str: Tensor of strings with location ID (e.g., "26")
        max_location_id: Maximum location ID for normalization (default 200)
    
    Returns:
        location_id: [B] or scalar, float32, normalized to [0, 1]
    """
    # Handle empty strings by replacing with "0"
    location_str = tf.where(
        tf.equal(tf.strings.length(location_str), 0),
        tf.fill(tf.shape(location_str), "0"),
        location_str
    )
    
    # Try to convert to number, default to 0.0 if it fails
    location_id = tf.strings.to_number(location_str, out_type=tf.float32)
    # Handle NaN or invalid values by setting to 0.0
    location_id = tf.where(tf.math.is_nan(location_id), tf.zeros_like(location_id), location_id)
    # Normalize to [0, 1]
    location_id = location_id / tf.cast(max_location_id, tf.float32)
    # Clip to [0, 1]
    location_id = tf.clip_by_value(location_id, 0.0, 1.0)
    return location_id


def encode_metadata_from_tfrecords(images, location_str, date_captured_str, max_location_id=200):
    """
    Encode metadata from TFRecords format with cyclical encoding.
    
    This function takes the raw strings from TFRecords and produces the encoded metadata tensor.
    
    Args:
        images: Image tensor [B, H, W, 3] in [0, 1] range
        location_str: Location string tensor [B] (e.g., "26")
        date_captured_str: Date string tensor [B] (e.g., "2013-10-04 13:31:53")
        max_location_id: Maximum location ID for normalization (default 200)
    
    Returns:
        metadata: Tensor of shape [B, 8] with cyclical encoding:
        [location_id, hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, brightness]
    """
    batch_size = tf.shape(images)[0]
    
    # Parse location
    location_id = parse_location_tf(location_str, max_location_id)
    
    # Parse date_captured to extract temporal features
    hour, day_of_week, month = parse_date_captured_tf(date_captured_str)
    
    # Convert to float32 for encoding
    hour = tf.cast(hour, tf.float32)
    day_of_week = tf.cast(day_of_week, tf.float32)
    month = tf.cast(month, tf.float32)
    
    # Compute brightness from image
    brightness = tf.reduce_mean(images, axis=[1, 2, 3])  # [B]
    
    # Encode with cyclical encoding
    metadata = encode_metadata_cyclical_tf(location_id, hour, day_of_week, month, brightness)
    
    return metadata


def create_placeholder_metadata_cyclical(batch_size, location_id_value=0.0, 
                                         hour_value=12.0, day_of_week_value=3.0, 
                                         month_value=6.0):
    """
    Create placeholder metadata with cyclical encoding for testing/placeholder use.
    
    Args:
        batch_size: Batch size (int or tensor)
        location_id_value: Location ID value (default 0.0)
        hour_value: Hour value (0-23, default 12.0 for noon)
        day_of_week_value: Day of week (0-6, default 3.0 for Thursday)
        month_value: Month value (1-12, default 6.0 for June)
    
    Returns:
        Tensor of shape [batch_size, 8] with cyclical encoding
    """
    location_id = tf.fill([batch_size], location_id_value)
    hour = tf.fill([batch_size], hour_value)
    day_of_week = tf.fill([batch_size], day_of_week_value)
    month = tf.fill([batch_size], month_value)
    brightness = tf.zeros([batch_size], dtype=tf.float32)  # Will be computed from image
    
    return encode_metadata_cyclical_tf(location_id, hour, day_of_week, month, brightness)

