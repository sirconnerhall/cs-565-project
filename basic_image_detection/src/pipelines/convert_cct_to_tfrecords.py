"""
Convert CCT dataset from JSON + image files to TFRecords format.

This script pre-processes the CCT dataset once, storing images and annotations
in TFRecords format for faster loading during training.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import tensorflow as tf
from PIL import Image

from .cct_pipeline import load_cct_annotations


def load_cct_splits(splits_path: str):
    """
    Load train/val/test splits from CaltechCameraTrapsSplits JSON file.
    
    Args:
        splits_path: Path to CaltechCameraTrapsSplits_v0.json
    
    Returns:
        dict with keys "train", "val", "test" (or similar), each containing
        a list of image IDs or filenames
    """
    splits_path = Path(splits_path)
    with open(splits_path, "r") as f:
        splits_data = json.load(f)
    
    # The structure may vary, but typically it's either:
    # 1. {"train": [image_ids], "val": [image_ids], ...}
    # 2. {"splits": {"train": [image_ids], "val": [image_ids], ...}}
    # 3. Or it might use filenames instead of IDs
    
    if "splits" in splits_data:
        splits = splits_data["splits"]
    else:
        splits = splits_data
    
    print(f"[CCT Splits] Loaded splits from {splits_path}")
    for split_name, split_list in splits.items():
        print(f"[CCT Splits]   {split_name}: {len(split_list)} items")
    
    return splits


def get_filenames_from_splits(splits_path: str, metadata_path: str, split_name: str):
    """
    Get list of filenames for a specific split by matching location IDs in metadata.
    
    The CaltechCameraTrapsSplits file contains location IDs (not image IDs), so we need
    to find all images that belong to those locations.
    
    Args:
        splits_path: Path to CaltechCameraTrapsSplits_v0.json
        metadata_path: Path to caltech_images_*.json (to map location IDs to filenames)
        split_name: Name of split to extract (e.g., "train", "val")
    
    Returns:
        set of filenames (relative paths) for the specified split
    """
    splits = load_cct_splits(splits_path)
    
    if split_name not in splits:
        # Try common variations
        split_variations = {
            "train": ["train", "training", "train_images"],
            "val": ["val", "validation", "val_images", "valid"],
            "test": ["test", "testing", "test_images"],
        }
        for key, variations in split_variations.items():
            if split_name == key:
                for var in variations:
                    if var in splits:
                        split_name = var
                        break
                break
        
        if split_name not in splits:
            raise ValueError(
                f"Split '{split_name}' not found in splits file. "
                f"Available splits: {list(splits.keys())}"
            )
    
    split_location_ids = splits[split_name]
    print(f"[CCT Splits] Found {len(split_location_ids)} location IDs in split '{split_name}'")
    
    # Debug: show first few items to understand the format
    if len(split_location_ids) > 0:
        sample_ids = split_location_ids[:3] if len(split_location_ids) >= 3 else split_location_ids
        print(f"[CCT Splits] Sample location IDs (first 3): {sample_ids}")
        print(f"[CCT Splits] Type of first item: {type(split_location_ids[0])}")
    
    # Convert location IDs to a set for fast lookup (handle both string and int)
    location_ids_set = set()
    for loc_id in split_location_ids:
        location_ids_set.add(str(loc_id))  # Normalize to string
        try:
            location_ids_set.add(int(loc_id))  # Also add as int if possible
        except (ValueError, TypeError):
            pass
    
    print(f"[CCT Splits] Looking for images in {len(location_ids_set)} unique locations")
    
    # Load metadata and find all images matching these location IDs
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    
    # Find all images that belong to the specified locations
    filenames = set()
    location_counts = {}
    
    for img in meta["images"]:
        img_location = img.get("location")
        if img_location is None:
            continue
        
        # Check if this image's location matches any in our split
        # Handle both string and int comparisons
        location_str = str(img_location)
        location_int = None
        try:
            location_int = int(img_location)
        except (ValueError, TypeError):
            pass
        
        if (location_str in location_ids_set or 
            location_int in location_ids_set or
            img_location in location_ids_set):
            filenames.add(img["file_name"])
            # Track counts for debugging
            loc_key = str(img_location)
            location_counts[loc_key] = location_counts.get(loc_key, 0) + 1
    
    print(f"[CCT Splits] Found {len(filenames)} images matching the location IDs")
    print(f"[CCT Splits] Images per location (sample): {dict(list(location_counts.items())[:5])}")
    
    if len(filenames) == 0:
        raise RuntimeError(
            f"Failed to find any images for split '{split_name}'. "
            f"Split had {len(split_location_ids)} location IDs, but 0 images were matched. "
            f"Check that the 'location' field in metadata matches the location IDs in the splits file."
        )
    
    return filenames


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_tf_example(sample: Dict, image_size: Optional[tuple] = None) -> tf.train.Example:
    """
    Convert a single CCT sample to a TFRecord Example.
    
    Args:
        sample: Dict with keys:
            - file_name: str
            - full_path: str
            - bboxes: list of [ymin, xmin, ymax, xmax] (normalized)
            - labels: list of int class indices
        image_size: Optional (H, W) tuple. If provided, images are resized.
                    If None, original size is preserved.
    
    Returns:
        tf.train.Example proto
    """
    full_path = Path(sample["full_path"])
    
    # Load and decode image
    try:
        with Image.open(full_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img, dtype=np.uint8)
            
            # Resize if requested
            if image_size is not None:
                img_pil = Image.fromarray(img_array)
                img_pil = img_pil.resize((image_size[1], image_size[0]), Image.Resampling.LANCZOS)
                img_array = np.array(img_pil, dtype=np.uint8)
    except Exception as e:
        raise RuntimeError(f"Failed to load image {full_path}: {e}")
    
    # Encode image as JPEG bytes (smaller than raw)
    # Convert numpy array to tensor for encoding
    img_tensor = tf.constant(img_array)
    img_bytes = tf.io.encode_jpeg(img_tensor, quality=95).numpy()
    
    # Get bboxes and labels
    bboxes = sample["bboxes"]
    labels = sample["labels"]
    
    # Convert to numpy arrays
    if len(bboxes) == 0:
        bboxes_np = np.zeros((0, 4), dtype=np.float32)
    else:
        bboxes_np = np.array(bboxes, dtype=np.float32)
    
    if len(labels) == 0:
        labels_np = np.zeros((0,), dtype=np.int64)
    else:
        labels_np = np.array(labels, dtype=np.int64)
    
    # Flatten arrays for TFRecord
    bboxes_flat = bboxes_np.flatten().tolist()
    labels_flat = labels_np.tolist()
    
    # Create feature dict
    feature = {
        'image/encoded': _bytes_feature(img_bytes),
        'image/format': _bytes_feature(b'jpeg'),
        'image/height': _int64_feature([img_array.shape[0]]),
        'image/width': _int64_feature([img_array.shape[1]]),
        'image/num_boxes': _int64_feature([len(bboxes)]),
        'bboxes': _float_feature(bboxes_flat),
        'labels': _int64_feature(labels_flat),
        'file_name': _bytes_feature(sample["file_name"].encode('utf-8')),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_cct_to_tfrecords(
    images_root: str,
    metadata_path: str,
    bboxes_path: str,
    output_dir: str,
    filelist_path: Optional[str] = None,
    split: str = "train",
    image_size: Optional[tuple] = None,
    num_shards: int = 10,
):
    """
    Convert CCT dataset to TFRecords format.
    
    Args:
        images_root: Root directory where images are stored
        metadata_path: Path to caltech_images_*.json
        bboxes_path: Path to caltech_bboxes_*.json
        output_dir: Directory to write TFRecords files
        filelist_path: Optional txt file with subset of filenames
        split: Split name (e.g., "train", "val")
        image_size: Optional (H, W) tuple to resize images. If None, preserves original size.
        num_shards: Number of shard files to create (for parallel loading)
    
    Returns:
        Path to the metadata JSON file containing category info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[CCT->TFRecords] Starting conversion for split: {split}")
    print(f"[CCT->TFRecords] Output directory: {output_dir}")
    print(f"[CCT->TFRecords] Image size: {image_size if image_size else 'original'}")
    print(f"[CCT->TFRecords] Number of shards: {num_shards}")
    
    # Load annotations (reuse existing logic)
    samples, categories = load_cct_annotations(
        metadata_path=metadata_path,
        bboxes_path=bboxes_path,
        images_root=images_root,
        filelist_path=filelist_path,
    )
    
    print(f"[CCT->TFRecords] Loaded {len(samples)} samples")
    
    # Save category metadata
    metadata_path = output_dir / f"{split}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({
            "categories": categories,
            "num_samples": len(samples),
            "image_size": image_size,
        }, f, indent=2)
    print(f"[CCT->TFRecords] Saved metadata to {metadata_path}")
    
    # Shuffle samples for better distribution across shards
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(len(samples))
    samples_shuffled = [samples[i] for i in indices]
    
    # Calculate samples per shard
    samples_per_shard = len(samples_shuffled) // num_shards
    if samples_per_shard == 0:
        samples_per_shard = 1
        num_shards = len(samples_shuffled)
    
    print(f"[CCT->TFRecords] Writing {num_shards} shard files...")
    
    # Write shards
    total_written = 0
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        if shard_idx == num_shards - 1:
            # Last shard gets all remaining samples
            end_idx = len(samples_shuffled)
        else:
            end_idx = (shard_idx + 1) * samples_per_shard
        
        shard_samples = samples_shuffled[start_idx:end_idx]
        
        shard_filename = f"{split}-{shard_idx:05d}-of-{num_shards:05d}.tfrecord"
        shard_path = output_dir / shard_filename
        
        with tf.io.TFRecordWriter(str(shard_path)) as writer:
            for sample in shard_samples:
                try:
                    example = create_tf_example(sample, image_size=image_size)
                    writer.write(example.SerializeToString())
                    total_written += 1
                except Exception as e:
                    print(f"[CCT->TFRecords] Warning: Failed to write sample {sample['file_name']}: {e}")
                    continue
        
        print(f"[CCT->TFRecords] Wrote shard {shard_idx + 1}/{num_shards}: {len(shard_samples)} samples -> {shard_path}")
    
    print(f"[CCT->TFRecords] Conversion complete!")
    print(f"[CCT->TFRecords] Total samples written: {total_written}/{len(samples)}")
    print(f"[CCT->TFRecords] Metadata saved to: {metadata_path}")
    
    return metadata_path


def convert_cct_splits_to_tfrecords(
    images_root: str,
    metadata_path: str,
    bboxes_path: str,
    output_dir: str,
    splits_path: Optional[str] = None,
    train_filelist_path: Optional[str] = None,
    val_filelist_path: Optional[str] = None,
    image_size: Optional[tuple] = None,
    train_num_shards: int = 10,
    val_num_shards: int = 5,
):
    """
    Convert both train and val splits to TFRecords.
    
    This is a convenience function that calls convert_cct_to_tfrecords for both splits.
    Can use either a splits JSON file or separate filelist txt files.
    
    Args:
        images_root: Root directory where images are stored
        metadata_path: Path to caltech_images_*.json
        bboxes_path: Path to caltech_bboxes_*.json
        output_dir: Directory to write TFRecords files
        splits_path: Path to CaltechCameraTrapsSplits_v0.json (preferred)
        train_filelist_path: Optional txt file with train split filenames (if splits_path not provided)
        val_filelist_path: Optional txt file with val split filenames (if splits_path not provided)
        image_size: Optional (H, W) tuple to resize images
        train_num_shards: Number of shard files for train split
        val_num_shards: Number of shard files for val split
    
    Returns:
        Tuple of (train_metadata_path, val_metadata_path)
    """
    print("=" * 60)
    print("Converting CCT dataset to TFRecords (train + val splits)")
    print("=" * 60)
    
    # Determine train/val filelists
    if splits_path:
        # Use splits JSON file
        print(f"[CCT Splits] Using splits file: {splits_path}")
        train_filenames = get_filenames_from_splits(splits_path, metadata_path, "train")
        val_filenames = get_filenames_from_splits(splits_path, metadata_path, "val")
        
        # Write temporary filelists (or pass as sets to load_cct_annotations)
        # For now, we'll create temporary filelist files
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(sorted(train_filenames)))
            train_filelist_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(sorted(val_filenames)))
            val_filelist_path = f.name
        
        print(f"[CCT Splits] Created temporary filelists from splits JSON")
    elif train_filelist_path and val_filelist_path:
        print(f"[CCT Splits] Using provided filelist files")
    else:
        raise ValueError(
            "Either splits_path or both train_filelist_path and val_filelist_path must be provided"
        )
    
    # Convert train split
    print("\n" + "=" * 60)
    print("Converting TRAIN split...")
    print("=" * 60)
    train_metadata = convert_cct_to_tfrecords(
        images_root=images_root,
        metadata_path=metadata_path,
        bboxes_path=bboxes_path,
        output_dir=output_dir,
        filelist_path=train_filelist_path,
        split="train",
        image_size=image_size,
        num_shards=train_num_shards,
    )
    
    # Convert val split
    print("\n" + "=" * 60)
    print("Converting VAL split...")
    print("=" * 60)
    val_metadata = convert_cct_to_tfrecords(
        images_root=images_root,
        metadata_path=metadata_path,
        bboxes_path=bboxes_path,
        output_dir=output_dir,
        filelist_path=val_filelist_path,
        split="val",
        image_size=image_size,
        num_shards=val_num_shards,
    )
    
    # Clean up temporary files if we created them
    if splits_path:
        import os
        try:
            os.unlink(train_filelist_path)
            os.unlink(val_filelist_path)
        except:
            pass
    
    print("\n" + "=" * 60)
    print("Conversion complete for both splits!")
    print("=" * 60)
    print(f"Train metadata: {train_metadata}")
    print(f"Val metadata: {val_metadata}")
    
    return train_metadata, val_metadata


def main():
    """CLI entry point for conversion script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert CCT dataset to TFRecords",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single split:
  python convert_cct_to_tfrecords.py --images_root D:/datasets/cct/images \\
      --metadata_path D:/datasets/cct/annotations/caltech_images_20210113.json \\
      --bboxes_path D:/datasets/cct/annotations/caltech_bboxes_20200316.json \\
      --output_dir D:/datasets/cct/tfrecords --split train \\
      --filelist_path D:/datasets/cct/train_files.txt --image_size 224 224

  # Convert both train and val splits (using splits JSON file - recommended):
  python convert_cct_to_tfrecords.py --images_root D:/datasets/cct/images \\
      --metadata_path D:/datasets/cct/annotations/caltech_images_20210113.json \\
      --bboxes_path D:/datasets/cct/annotations/caltech_bboxes_20200316.json \\
      --output_dir D:/datasets/cct/tfrecords \\
      --splits_path D:/datasets/cct/annotations/CaltechCameraTrapsSplits_v0.json \\
      --image_size 224 224

  # Or using separate filelist files:
  python convert_cct_to_tfrecords.py --images_root D:/datasets/cct/images \\
      --metadata_path D:/datasets/cct/annotations/caltech_images_20210113.json \\
      --bboxes_path D:/datasets/cct/annotations/caltech_bboxes_20200316.json \\
      --output_dir D:/datasets/cct/tfrecords \\
      --train_filelist D:/datasets/cct/train_files.txt \\
      --val_filelist D:/datasets/cct/val_files.txt \\
      --image_size 224 224
        """
    )
    parser.add_argument("--images_root", type=str, required=True,
                        help="Root directory where images are stored")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="Path to caltech_images_*.json")
    parser.add_argument("--bboxes_path", type=str, required=True,
                        help="Path to caltech_bboxes_*.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write TFRecords files")
    
    # Options for single split conversion
    parser.add_argument("--filelist_path", type=str, default=None,
                        help="Optional txt file with subset of filenames (for single split)")
    parser.add_argument("--split", type=str, default=None,
                        help="Split name (train, val, etc.) - for single split conversion")
    
    # Options for both splits conversion
    parser.add_argument("--splits_path", type=str, default=None,
                        help="Path to CaltechCameraTrapsSplits_v0.json (preferred over filelists)")
    parser.add_argument("--train_filelist", type=str, default=None,
                        help="Txt file with train split filenames (if splits_path not provided)")
    parser.add_argument("--val_filelist", type=str, default=None,
                        help="Txt file with val split filenames (if splits_path not provided)")
    
    parser.add_argument("--image_size", type=int, nargs=2, default=None,
                        help="Optional image size as H W (e.g., 224 224). If not provided, preserves original size.")
    parser.add_argument("--num_shards", type=int, default=10,
                        help="Number of shard files to create (for single split)")
    parser.add_argument("--train_num_shards", type=int, default=10,
                        help="Number of shard files for train split (when converting both)")
    parser.add_argument("--val_num_shards", type=int, default=5,
                        help="Number of shard files for val split (when converting both)")
    
    args = parser.parse_args()
    
    image_size = tuple(args.image_size) if args.image_size else None
    
    # Determine if we should convert both splits
    convert_both = args.splits_path is not None or (args.train_filelist is not None or args.val_filelist is not None)
    
    if convert_both:
        # Convert both splits
        if args.splits_path is None:
            if args.train_filelist is None or args.val_filelist is None:
                parser.error("Either --splits_path or both --train_filelist and --val_filelist must be provided")
        
        convert_cct_splits_to_tfrecords(
            images_root=args.images_root,
            metadata_path=args.metadata_path,
            bboxes_path=args.bboxes_path,
            output_dir=args.output_dir,
            splits_path=args.splits_path,
            train_filelist_path=args.train_filelist,
            val_filelist_path=args.val_filelist,
            image_size=image_size,
            train_num_shards=args.train_num_shards,
            val_num_shards=args.val_num_shards,
        )
    else:
        # Convert single split
        if args.split is None:
            args.split = "train"  # Default to train
        
        convert_cct_to_tfrecords(
            images_root=args.images_root,
            metadata_path=args.metadata_path,
            bboxes_path=args.bboxes_path,
            output_dir=args.output_dir,
            filelist_path=args.filelist_path,
            split=args.split,
            image_size=image_size,
            num_shards=args.num_shards,
        )


if __name__ == "__main__":
    main()

