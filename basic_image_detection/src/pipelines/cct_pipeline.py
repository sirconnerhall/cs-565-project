import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image


# ----------------------------------------------------
# Info objects that mimic TFDS's ds_info.features API
# ----------------------------------------------------

class CCTLabelInfo:
    """
    Mimic TFDS info.features["objects"]["label"] for compatibility.
    """
    def __init__(self, categories):
        # categories: list of {"id": int, "name": str}
        # Sort by original category_id so indices are consistent and stable.
        self._cats = sorted(categories, key=lambda c: c["id"])
        self.names = [c["name"] for c in self._cats]
        self.num_classes = len(self.names)


class CCTInfo:
    """
    Minimal info object with a .features["objects"]["label"] member.
    """
    def __init__(self, categories):
        self.features = {
            "objects": {
                "label": CCTLabelInfo(categories)
            }
        }


# ----------------------------------------------------
# Load annotations & build sample list
# ----------------------------------------------------

def load_cct_annotations(
    metadata_path,
    bboxes_path,
    images_root,
    filelist_path=None,
    filter_empty=False,
):
    """
    metadata_path: path to caltech_camera_traps.json (image metadata)
    bboxes_path:   path to caltech_bboxes_20200316.json (categories + bbox annotations)
    images_root:   directory where images are stored
    filelist_path: optional txt with one file_name per line to restrict to a subset
                   (e.g., your 5% sample).
    filter_empty: if True, exclude images with no bbox annotations (helps with class imbalance)

    Returns:
        samples: list of dicts:
            - file_name   (relative path)
            - full_path   (absolute path on disk)
            - bboxes: list of [ymin, xmin, ymax, xmax] (normalized)
            - labels: list of int class indices (0..C-1)
        categories_sorted: list of category dicts (sorted by original id)
    """

    metadata_path = Path(metadata_path)
    bboxes_path = Path(bboxes_path)
    images_root = Path(images_root)

    # 1) Load metadata (images + maybe categories, but we don't rely on its categories)
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    meta_images = meta["images"]  # list of {id, file_name, width, height, ...}
    print(f"[CCT] Loaded metadata for {len(meta_images)} images from {metadata_path}")

    # Map image_id -> (file_name, width, height, metadata)
    imginfo_by_id = {}
    for img in meta_images:
        img_id = img["id"]
        imginfo_by_id[img_id] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "location": img.get("location"),
            "date_captured": img.get("date_captured"),
            "seq_id": img.get("seq_id"),
        }

    # 2) Load bbox annotations and categories
    with open(bboxes_path, "r") as f:
        bbox_data = json.load(f)

    categories = bbox_data["categories"]     # [{"id": int, "name": str}, ...]
    annotations = bbox_data["annotations"]   # [{"image_id": ..., "bbox": [...], "category_id": ...}, ...]

    print(f"[CCT] Loaded {len(categories)} categories and {len(annotations)} bbox annotations from {bboxes_path}")

    # Sort categories by original id and build mapping from original category_id -> contiguous index
    categories_sorted = sorted(categories, key=lambda c: c["id"])
    catid_to_index = {c["id"]: i for i, c in enumerate(categories_sorted)}

    # 3) Optional: restrict by file list (5% sample)
    allowed_files = None
    if filelist_path is not None:
        filelist_path = Path(filelist_path)
        with open(filelist_path, "r") as f:
            allowed_files = set(line.strip() for line in f if line.strip())
        print(f"[CCT] Using filelist {filelist_path}, {len(allowed_files)} filenames listed")

    # 4) Build image_id -> list of annotations
    anns_by_image = {}
    for ann in annotations:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    # 5) Build samples
    samples = []
    num_missing_files = 0
    num_missing_meta = 0
    num_no_ann = 0

    # We iterate over *metadata* images so we can include empty images too
    for img_id, info in imginfo_by_id.items():
        file_name = info["file_name"]
        width = info["width"]
        height = info["height"]

        if allowed_files is not None and file_name not in allowed_files:
            continue

        full_path = images_root / file_name
        if not full_path.exists():
            num_missing_files += 1
            continue
        
        # Quick validation: try to actually decode the image file to check if it's valid
        # This catches corrupted files before they reach TensorFlow
        # Note: We do a full decode (not just verify) because verify() only checks headers
        # and TensorFlow's decoder is stricter and will fail on files that pass verify()
        try:
            # Actually decode the image to catch issues TensorFlow will find
            with Image.open(full_path) as img:
                img.load()  # Actually load/decode the image (more thorough than verify())
                # Quick check that it can be converted to RGB
                if img.mode != 'RGB':
                    img.convert('RGB')
        except Exception as e:
            # File exists but is corrupted/invalid - skip it
            num_missing_files += 1
            continue

        anns = anns_by_image.get(img_id, [])

        bboxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]  # [x, y, w, h] in absolute pixels
            ymin = y / height
            xmin = x / width
            ymax = (y + h) / height
            xmax = (x + w) / width

            # Safety clip
            ymin = max(0.0, min(1.0, ymin))
            xmin = max(0.0, min(1.0, xmin))
            ymax = max(0.0, min(1.0, ymax))
            xmax = max(0.0, min(1.0, xmax))

            if ymax <= ymin or xmax <= xmin:
                continue  # skip degenerate boxes

            orig_cat = ann["category_id"]
            if orig_cat not in catid_to_index:
                # Unknown category id; skip
                continue
            class_idx = catid_to_index[orig_cat]

            bboxes.append([ymin, xmin, ymax, xmax])
            labels.append(class_idx)

        if len(anns) == 0:
            num_no_ann += 1
            # Skip empty images if filter_empty is True
            if filter_empty:
                continue

        samples.append(
            {
                "file_name": file_name,
                "full_path": str(full_path),
                "bboxes": bboxes,
                "labels": labels,
                "location": info.get("location"),
                "date_captured": info.get("date_captured"),
                "seq_id": info.get("seq_id"),
            }
        )

    print(f"[CCT] Built {len(samples)} samples from metadata + bbox annotations")
    print(f"[CCT]   Missing/corrupted image files: {num_missing_files}")
    print(f"[CCT]   Images with no bbox annotations: {num_no_ann}")

    if len(samples) == 0:
        raise RuntimeError(
            "[CCT] No samples found. "
            "Check that images_root is correct, filelist filenames "
            "match the JSON 'file_name' fields, and that images exist."
        )

    return samples, categories_sorted


# ----------------------------------------------------
# Extract CCT metadata features
# ----------------------------------------------------

def extract_cct_metadata_features(sample):
    """
    Extract metadata features from a CCT sample with cyclical encoding.
    
    Args:
        sample: Dict with keys: location, date_captured, seq_id
    
    Returns:
        numpy array of metadata features (8 features):
        - location_id (normalized)
        - hour_sin, hour_cos (cyclical encoding)
        - day_of_week_sin, day_of_week_cos (cyclical encoding)
        - month_sin, month_cos (cyclical encoding)
        - brightness (will be computed from image, placeholder 0 here)
    """
    from ..utils.metadata_encoding import encode_metadata_cyclical_numpy
    
    location = sample.get("location")
    date_str = sample.get("date_captured")
    
    # Location ID (normalize to 0-1, assuming max location ID ~200)
    location_id = 0.0
    if location is not None:
        try:
            loc_int = int(location) if isinstance(location, str) else location
            location_id = float(loc_int) / 200.0  # Normalize
        except (ValueError, TypeError):
            pass
    
    # Date/time features
    hour = 0.0
    day_of_week = 0.0
    month = 0.0
    
    if date_str:
        try:
            # Parse date string (format: "2013-10-04 13:31:53")
            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            hour = dt.hour  # Keep as 0-23 for cyclical encoding
            day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
            month = dt.month  # 1-12
        except (ValueError, TypeError):
            pass
    
    # Brightness placeholder (will be computed from image in dataset pipeline)
    brightness = 0.0
    
    # Use cyclical encoding
    return encode_metadata_cyclical_numpy(location_id, hour, day_of_week, month, brightness)


def add_cct_metadata(image, targets, sample_metadata):
    """
    Add CCT metadata to dataset pipeline with cyclical encoding.
    Computes brightness from image and adds to metadata.
    
    Args:
        image: [H, W, 3] float32 image in [0, 1]
        targets: Dict with bboxes and labels
        sample_metadata: Dict with location, date_captured, etc.
    
    Returns:
        (image, metadata), targets where metadata is [8] with cyclical encoding
    """
    from ..utils.metadata_encoding import encode_metadata_cyclical_tf
    
    # Compute brightness from image
    brightness = tf.reduce_mean(image)  # [B] or scalar
    
    # Extract other metadata features
    location = sample_metadata.get("location", None)
    date_str = sample_metadata.get("date_captured", None)
    
    # Location ID
    location_id = 0.0
    if location is not None:
        try:
            loc_int = int(location) if isinstance(location, str) else location
            location_id = tf.constant(float(loc_int) / 200.0, dtype=tf.float32)
        except (ValueError, TypeError):
            location_id = tf.constant(0.0, dtype=tf.float32)
    else:
        location_id = tf.constant(0.0, dtype=tf.float32)
    
    # Date/time features (simplified - would need proper parsing in TF)
    # For now, use placeholders that will be computed from date string
    # In practice, you'd parse this in the dataset pipeline
    # Use actual hour values (0-23) for cyclical encoding, not normalized
    hour = tf.constant(12.0, dtype=tf.float32)  # Placeholder: noon
    day_of_week = tf.constant(3.0, dtype=tf.float32)  # Placeholder: Thursday
    month = tf.constant(6.0, dtype=tf.float32)  # Placeholder: June
    
    # Use cyclical encoding
    metadata = encode_metadata_cyclical_tf(location_id, hour, day_of_week, month, brightness)
    
    return (image, metadata), targets


# ----------------------------------------------------
# Make tf.data.Dataset
# ----------------------------------------------------

def make_cct_dataset(
    images_root,
    metadata_path,
    bboxes_path,
    filelist_path=None,
    split="train",      # currently only used for logging
    batch_size=8,
    image_size=(224, 224),
    shuffle=True,
    filter_empty=False,
):
    """
    Build a dataset similar to make_coco_dataset:

        (images, {"bboxes": ..., "labels": ...})

    - images_root: root path where images live (e.g. D:/datasets/cct/images)
    - metadata_path: full path to caltech_camera_traps.json
    - bboxes_path:   full path to caltech_bboxes_20200316.json
    - filelist_path: optional txt file listing the subset of filenames you actually downloaded
    - split: only used for logging at the moment; you can later create separate filelists per split.
    """

    print(f"[CCT] make_cct_dataset(split={split}, batch_size={batch_size}, image_size={image_size})")

    samples, categories = load_cct_annotations(
        metadata_path=metadata_path,
        bboxes_path=bboxes_path,
        images_root=images_root,
        filelist_path=filelist_path,
        filter_empty=filter_empty,
    )

    image_paths = [s["full_path"] for s in samples]
    bboxes_list = [s["bboxes"] for s in samples]
    labels_list = [s["labels"] for s in samples]

    def _gen():
        for p, b, l in zip(image_paths, bboxes_list, labels_list):
            # Convert to numpy arrays with correct shapes
            # Empty bboxes should be (0, 4) not (0,)
            if len(b) == 0:
                b = np.zeros((0, 4), dtype=np.float32)
            else:
                b = np.array(b, dtype=np.float32)
            
            # Empty labels should be (0,)
            if len(l) == 0:
                l = np.zeros((0,), dtype=np.int32)
            else:
                l = np.array(l, dtype=np.int32)
            
            yield p, b, l

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )

    def _parse_fn(path, bboxes, labels):
        """
        Parse image file with robust error handling.
        Uses PIL to decode (more robust than TensorFlow's decoder) then converts to tensor.
        """
        def _decode_with_pil(path_bytes):
            """
            Decode image using PIL (more robust) and return as numpy array.
            Returns black image if decode fails.
            """
            path_str = path_bytes.numpy().decode('utf-8')
            try:
                # Use PIL to decode - it's more robust than TensorFlow's decoder
                img = Image.open(path_str)
                # Convert to RGB if needed (handles grayscale, RGBA, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Convert to numpy array
                img_array = np.array(img, dtype=np.uint8)
                return img_array, True
            except Exception:
                # If decode fails, return black image
                return np.zeros((*image_size, 3), dtype=np.uint8), False
        
        # Use py_function to handle Python exceptions properly
        image_np, is_valid = tf.py_function(
            func=_decode_with_pil,
            inp=[path],
            Tout=[tf.uint8, tf.bool]
        )
        
        # Set shape
        image_np.set_shape([None, None, 3])
        is_valid.set_shape([])
        
        # Resize and normalize
        image = tf.image.resize(image_np, image_size)
        image = tf.cast(image, tf.float32) / 255.0

        return image, {
            "bboxes": bboxes,
            "labels": labels,
        }, is_valid

    ds = ds.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Filter out samples where image decoding failed
    def _filter_valid(images, targets, is_valid):
        return is_valid
    
    def _remove_valid_flag(images, targets, is_valid):
        return images, targets
    
    ds = ds.filter(_filter_valid)
    ds = ds.map(_remove_valid_flag, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1024)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            [*image_size, 3],
            {
                "bboxes": [None, 4],
                "labels": [None],
            },
        ),
        padding_values=(
            0.0,
            {
                "bboxes": tf.constant(0.0, dtype=tf.float32),
                "labels": tf.constant(0, dtype=tf.int32),
            },
        ),
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    info = CCTInfo(categories)
    return ds, info
