"""
Load CCT dataset from TFRecords format.

This provides a fast loading pipeline similar to COCO's TFDS approach,
but for the CCT dataset stored in TFRecords format.
"""

import json
from pathlib import Path

import tensorflow as tf

from cct_pipeline import CCTInfo, CCTLabelInfo


def parse_tf_example(example_proto, image_size=None):
    """
    Parse a TFRecord Example proto.
    
    Args:
        example_proto: Serialized tf.train.Example
        image_size: Optional (H, W) tuple. If provided, images are resized.
                    If None, uses original size from TFRecord.
    
    Returns:
        (image, targets_dict, is_valid)
        - image: [H, W, 3] float32 in [0, 1]
        - targets_dict: {"bboxes": [N, 4], "labels": [N]}
        - is_valid: bool scalar
    """
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/num_boxes': tf.io.FixedLenFeature([], tf.int64),
        'bboxes': tf.io.VarLenFeature(tf.float32),
        'labels': tf.io.VarLenFeature(tf.int64),
        'file_name': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode image
    image_bytes = parsed['image/encoded']
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    
    # Resize if requested (image_size should always be set by caller, but check for safety)
    if image_size is not None:
        image = tf.image.resize(image, image_size)
    
    # Normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    # Parse bboxes and labels
    num_boxes = tf.cast(parsed['image/num_boxes'], tf.int32)
    
    bboxes_flat = tf.sparse.to_dense(parsed['bboxes'])
    bboxes = tf.reshape(bboxes_flat, [num_boxes, 4])
    
    labels_flat = tf.sparse.to_dense(parsed['labels'])
    labels = tf.cast(labels_flat, tf.int32)
    
    # Handle empty case
    bboxes = tf.cond(
        num_boxes > 0,
        lambda: bboxes,
        lambda: tf.zeros((0, 4), dtype=tf.float32)
    )
    labels = tf.cond(
        num_boxes > 0,
        lambda: labels,
        lambda: tf.zeros((0,), dtype=tf.int32)
    )
    
    targets = {
        "bboxes": bboxes,
        "labels": labels,
    }
    
    return image, targets


def load_tfrecords_metadata(metadata_path):
    """
    Load category metadata from the JSON file created during conversion.
    
    Args:
        metadata_path: Path to *_metadata.json file
    
    Returns:
        categories: list of category dicts
        num_samples: int
        image_size: tuple or None
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return (
        metadata["categories"],
        metadata.get("num_samples", 0),
        metadata.get("image_size"),
    )


def make_cct_tfrecords_dataset(
    tfrecords_dir,
    metadata_path=None,
    split="train",
    batch_size=8,
    image_size=(224, 224),
    shuffle=True,
    num_parallel_calls=tf.data.AUTOTUNE,
):
    """
    Build a dataset from TFRecords, similar to make_coco_dataset.
    
    Args:
        tfrecords_dir: Directory containing TFRecord shard files
        metadata_path: Path to *_metadata.json file. If None, will try to find it.
        split: Split name (e.g., "train", "val")
        batch_size: Batch size
        image_size: (H, W) tuple for resizing images
        shuffle: Whether to shuffle the dataset
        num_parallel_calls: Number of parallel calls for map operations
    
    Returns:
        (dataset, info) where:
        - dataset: tf.data.Dataset yielding (images, {"bboxes": ..., "labels": ...})
        - info: CCTInfo object with category information
    """
    tfrecords_dir = Path(tfrecords_dir)
    
    print(f"[CCT-TFRecords] Loading dataset from {tfrecords_dir}")
    print(f"[CCT-TFRecords] Split: {split}, batch_size: {batch_size}, image_size: {image_size}")
    
    # Find TFRecord files
    pattern = str(tfrecords_dir / f"{split}-*-of-*.tfrecord")
    tfrecord_files = tf.io.matching_files(pattern).numpy()
    tfrecord_files = [f.decode('utf-8') for f in tfrecord_files]
    
    if len(tfrecord_files) == 0:
        raise FileNotFoundError(
            f"No TFRecord files found matching pattern: {pattern}\n"
            f"Make sure you've run convert_cct_to_tfrecords.py first."
        )
    
    print(f"[CCT-TFRecords] Found {len(tfrecord_files)} shard files")
    
    # Load metadata
    if metadata_path is None:
        metadata_path = tfrecords_dir / f"{split}_metadata.json"
    
    if not Path(metadata_path).exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Make sure you've run convert_cct_to_tfrecords.py first."
        )
    
    categories, num_samples, stored_image_size = load_tfrecords_metadata(metadata_path)
    print(f"[CCT-TFRecords] Loaded metadata: {len(categories)} categories, {num_samples} samples")
    
    # Use stored image size if available and no explicit size requested
    # (This allows loading pre-resized images without re-resizing)
    if stored_image_size is not None and image_size is None:
        image_size = tuple(stored_image_size)
        print(f"[CCT-TFRecords] Using stored image size: {image_size}")
    
    # Ensure image_size is set (required for batching)
    if image_size is None:
        raise ValueError(
            "image_size must be provided or stored in metadata. "
            "Either pass image_size parameter or convert with --image_size."
        )
    
    # Create dataset from TFRecord files
    ds = tf.data.TFRecordDataset(tfrecord_files)
    
    # Parse examples
    parse_fn = lambda x: parse_tf_example(x, image_size=image_size)
    ds = ds.map(parse_fn, num_parallel_calls=num_parallel_calls)
    
    # Shuffle
    if shuffle:
        ds = ds.shuffle(1024)
    
    # Batch with padding
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            [*image_size, 3],  # image
            {
                "bboxes": [None, 4],  # variable number of boxes
                "labels": [None],
            },
        ),
        padding_values=(
            0.0,  # image padding value
            {
                "bboxes": tf.constant(0.0, dtype=tf.float32),
                "labels": tf.constant(0, dtype=tf.int32),
            },
        ),
    )
    
    # Prefetch
    ds = ds.prefetch(num_parallel_calls)
    
    # Create info object
    info = CCTInfo(categories)
    
    return ds, info


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cct_tfrecords_pipeline.py <tfrecords_dir> [split]")
        sys.exit(1)
    
    tfrecords_dir = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    
    ds, info = make_cct_tfrecords_dataset(
        tfrecords_dir=tfrecords_dir,
        split=split,
        batch_size=4,
        image_size=(224, 224),
    )
    
    print(f"\nDataset info:")
    print(f"  Num classes: {info.features['objects']['label'].num_classes}")
    print(f"  Class names: {info.features['objects']['label'].names[:5]}...")
    
    print(f"\nTesting dataset...")
    for batch_images, batch_targets in ds.take(1):
        print(f"  Images shape: {batch_images.shape}")
        print(f"  Bboxes shape: {batch_targets['bboxes'].shape}")
        print(f"  Labels shape: {batch_targets['labels'].shape}")
        break
    
    print("✓ Dataset loading works!")

