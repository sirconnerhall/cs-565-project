"""
Train all four detector models successively:
- SSNM: Single Stage, No Metadata
- SSM: Single Stage, with Metadata
- TSNM: Two Stage, No Metadata
- TSM: Two Stage, with Metadata

All models use:
- Anchor-based detection (configurable via num_anchors in config.json)
- EfficientNet-B0 or CSPDarkNet backbone (configurable via pretrained_model_type)
- Focal loss for object detection

Configuration is read from configs/config.json.
"""

import sys
import traceback
from pathlib import Path
import time
import subprocess
import os

# Get project root
project_root = Path(__file__).resolve().parent.parent

def train_model(model_name, module_path):
    """Train a model by running it as a subprocess."""
    print("\n" + "=" * 80)
    print(f"TRAINING {model_name}")
    print("=" * 80)
    try:
        # Run as module to handle relative imports correctly
        # Output will be shown in real-time
        result = subprocess.run(
            [sys.executable, "-m", module_path],
            cwd=str(project_root),
            check=False
        )
        
        if result.returncode == 0:
            print(f"\n✓ {model_name} training completed successfully")
            return True
        else:
            print(f"\n✗ {model_name} training failed with exit code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n✗ {model_name} training failed: {e}")
        traceback.print_exc()
        return False


def train_ssnm():
    """Train SSNM model."""
    return train_model("SSNM (Single Stage, No Metadata)", "src.SSNM.train_detector")


def train_ssm():
    """Train SSM model."""
    return train_model("SSM (Single Stage, with Metadata)", "src.SSM.train_detector")


def train_tsnm():
    """Train TSNM model."""
    return train_model("TSNM (Two Stage, No Metadata)", "src.TSNM.train_detector")


def train_tsm():
    """Train TSM model."""
    return train_model("TSM (Two Stage, with Metadata)", "src.TSM.train_detector")


def main():
    """Train all models in sequence."""
    print("=" * 80)
    print("TRAINING ALL MODELS")
    print("=" * 80)
    print("\nThis will train all 4 models in sequence:")
    print("  1. SSNM - Single Stage, No Metadata")
    print("  2. SSM  - Single Stage, with Metadata")
    print("  3. TSNM - Two Stage, No Metadata")
    print("  4. TSM  - Two Stage, with Metadata")
    print("\nArchitecture:")
    print("  - Anchor-based detection (YOLO-style)")
    print("  - Backbone: EfficientNet-B0 (or CSPDarkNet)")
    print("  - All configuration from configs/config.json")
    print("\nNote: This may take a long time depending on your configuration.")
    
    start_time = time.time()
    results = {}
    
    # Train each model
    models = [
        ("SSNM", train_ssnm),
        ("SSM", train_ssm),
        ("TSNM", train_tsnm),
        ("TSM", train_tsm),
    ]
    
    for model_name, train_func in models:
        model_start = time.time()
        success = train_func()
        model_time = time.time() - model_start
        
        results[model_name] = {
            "success": success,
            "time_seconds": model_time,
            "time_minutes": model_time / 60,
        }
        
        if success:
            print(f"\n{model_name} completed in {model_time/60:.2f} minutes")
        else:
            print(f"\n{model_name} failed after {model_time/60:.2f} minutes")
        
        # Small pause between models
        if model_name != models[-1][0]:  # Don't pause after last model
            print("\n" + "-" * 80)
            print("Preparing next model...")
            time.sleep(2)
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    successful = [name for name, r in results.items() if r["success"]]
    failed = [name for name, r in results.items() if not r["success"]]
    
    print(f"\nTotal time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    if successful:
        print(f"  ✓ {', '.join(successful)}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        print(f"  ✗ {', '.join(failed)}")
    
    print("\n" + "-" * 80)
    print("Individual model times:")
    for model_name, r in results.items():
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {model_name}: {r['time_minutes']:.2f} minutes")
    
    print("\n" + "=" * 80)
    
    if failed:
        print("\nWarning: Some models failed to train. Check the error messages above.")
        return 1
    else:
        print("\nAll models trained successfully!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

