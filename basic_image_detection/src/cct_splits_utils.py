"""
Utility functions for working with CCT splits file.
"""

import tempfile
from pathlib import Path
from typing import Optional


def get_filelist_from_splits_or_config(
    config: dict,
    split_name: str,
    metadata_path: str,
) -> Optional[str]:
    """
    Get filelist path for a split, using explicit filelist if provided,
    otherwise generating from splits file.
    
    Args:
        config: Configuration dictionary
        split_name: "train" or "val"
        metadata_path: Path to CCT metadata JSON file
        
    Returns:
        Path to filelist file (or None if using TFRecords)
    """
    # Check for explicit filelist first
    if split_name == "train":
        filelist = config.get("cct_train_file")
    elif split_name == "val":
        filelist = config.get("cct_val_file")
    else:
        filelist = None
    
    if filelist:
        return filelist
    
    # If no explicit filelist, try to generate from splits file
    splits_path = config.get("cct_splits")
    if splits_path and Path(splits_path).exists():
        from convert_cct_to_tfrecords import get_filenames_from_splits
        
        print(f"[CCT Splits] Generating {split_name} filelist from splits file: {splits_path}")
        filenames = get_filenames_from_splits(splits_path, metadata_path, split_name)
        
        # Create temporary filelist file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('\n'.join(sorted(filenames)))
            temp_filelist = f.name
        
        print(f"[CCT Splits] Created temporary filelist: {temp_filelist}")
        return temp_filelist
    
    # No filelist and no splits file - return None (will use all images)
    return None

