

# **Basic Image Detection – Multimodal (Image + Metadata) Pipeline**

This project implements a complete **end-to-end image classification training pipeline** using:

* **TensorFlow 2.10 GPU on Windows**
* **COCO 2017 dataset** (via **TensorFlow Datasets**)
* **Multilabel classification** (sigmoid outputs)
* **Focal Loss** (for imbalanced classes)
* **MobileNetV2 backbone** (transfer learning)
* **Backbone fine-tuning**
* **Synthetic metadata** + **multimodal fusion model**

The goal of this project is to develop a flexible training pipeline that supports **transfer learning**, **metadata integration**, and **eventually the Caltech Camera Traps dataset** for research on context-aware detection.

---

# Table of Contents

1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [COCO Dataset Setup](#coco-dataset-setup)
4. [Training Scripts](#training-scripts)

   * Single-input multilabel model
   * Multimodal image+metadata model (current)
5. [Synthetic Metadata](#synthetic-metadata)
6. [Evaluation Scripts](#evaluation-scripts)
7. [Prediction Script](#prediction-script)
8. [Known Limitations](#known-limitations)
9. [Next Steps: Camera Traps Dataset](#next-steps-camera-traps-dataset)

---

# **Project Structure**

```
basic_image_detection/
├── configs/
│   └── coco_multilabel_config.json
├── models/
│   └── (saved .keras model files: *_best.keras and *_last.keras)
├── src/
│   ├── coco_tfds_pipeline.py                # COCO TFDS loader (images + bbox + labels)
│   ├── train_coco_multilabel.py             # Older single-input classification trainer
│   ├── train_coco_multimodal_synthmeta.py   # CURRENT multimodal trainer (image+metadata)
│   ├── eval_coco_multilabel.py              # Single-input eval script
│   ├── eval_coco_multimodal_synthmeta.py    # CURRENT multimodal eval script
│   ├── visualize_coco_batch.py              # Visualize raw COCO annotations + images
└── requirements.txt
```

---

# **Environment Setup**

This project runs on **TensorFlow 2.10.1 GPU** on **Windows**.

TF 2.10.1 is the *last* version that supports GPU on Windows.

### Python version

Use **Python 3.10.x**

### Install CUDA & cuDNN

TF 2.10.1 requires:

* **CUDA 11.2 – 11.8**
* **cuDNN 8.x**

After installation, ensure:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
```

is on your **PATH**.

### Create venv

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU test

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

Expected output includes:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

# **COCO Dataset Setup**

This project uses **TensorFlow Datasets (TFDS)** to load **COCO 2017**.

Set the TFDS download directory (to avoid filling the C: drive):

```python
os.environ["TFDS_DATA_DIR"] = "D:/tensorflow_datasets"
```

Then the scripts automatically download or read COCO data from:

```
D:/tensorflow_datasets/coco/2017/1.1.0/
```

This directory also contains:

* `object_label.labels` → **80 COCO class names** (used by prediction script)

---

# **Training Scripts**

## 1. **`train_coco_multilabel.py` (older)**

A simple MobileNetV2 multilabel classifier:

* Image-only
* Freeze then unfreeze backbone
* Binary cross entropy or focal loss
* No metadata

Kept for reference; replaced by the multimodal script below.

---

## 2. **`train_coco_multimodal_synthmeta.py` (CURRENT)**

This is the main training script.

### Input Format

The dataset pipeline emits:

```
((image_tensor, metadata_vector), multi_hot_label)
```

Where:

* `image_tensor`: `(B, H, W, 3)` float32, resized + normalized
* `metadata_vector`: synthetic metadata (dim=7)
* `multi_hot_label`: 80-length vector (COCO multilabel)

### Model Architecture

#### Image branch

* MobileNetV2 (ImageNet weights)
* Global pooling
* Dropout

#### Metadata branch

* Dense(32) → Dense(16)

#### Fusion

* Concatenate(image_features, metadata_features)
* Dense(128) → Dropout → Dense(80, sigmoid)

### Training Phases

**Phase 1** (warm-up):

* Backbone frozen
* Train classification & metadata head

**Phase 2** (fine-tuning):

* Unfreeze last X% of backbone
* Use smaller LR
* Continue training

### Loss Function

Uses **binary focal loss**:

* `gamma=2`
* `alpha=0.25`
* Optional `label_smoothing`

This helps reduce class imbalance effects (e.g., “person” dominating).

### Saving Models

Two files saved:

```
models/<modelname>_best.keras
models/<modelname>_last.keras
```

---

# **Synthetic Metadata**

Used to develop and test the **multimodal architecture** before switching to real metadata.

For each image:

1. **brightness** = mean pixel intensity
2. **is_day / is_night** (based on brightness threshold)
3. **location_onehot(4)** = 4 synthetic location categories

Metadata vector is:

```
[brightness, is_day, is_night, loc0, loc1, loc2, loc3]
```

Total dimension: **7**

This metadata is computationally cheap and sufficient to develop and validate the multimodal pipeline.

---

# **Evaluation Scripts**

## 1. **`eval_coco_multimodal_synthmeta.py` (CURRENT)**

Evaluates multimodal model:

* Loads either best or last `.keras`
* Loads COCO validation split
* Applies **multilabel** + **synthetic metadata** transforms
* Recompiles with BCE (no need for focal during eval)
* Computes:

  * `binary_accuracy`
  * `AUC`

Also prints example predictions vs GT:

```
GT labels: [...]
Pred labels: [...]
```

## 2. **`eval_coco_multilabel.py` (older)**

Used for image-only model.
Not compatible with multimodal input.

---

# **Known Limitations**

### 1. **COCO is not a good dataset for multilabel training**

Because:

* Very imbalanced (“person” dominates)
* Many images with multiple overlapping categories
* Designed for detection, not image-level classification

This caused persistent “person-only predictions” even with focal loss.

### 2. Metadata is synthetic (temporary)

For true multimodal benefits, real metadata is needed.

### 3. No bounding box detection yet

Current pipeline is multilabel *classification*, not object detection.

---

# **Next Steps: Move to Caltech Camera Traps**

The next logical step is switching datasets.

### Why Camera Traps?

From Caltech Camera Traps (CCT):

* 21 species + empty
* 66k bounding boxes
* Images grouped by **camera location**
* Camera/location/time metadata available
* COCO-style annotations
* Designed for research on **generalization across locations**

This fits your project goal *perfectly*.

### What changes when switching?

You will need:

1. A new dataset loader:

   * Load images from disk
   * Parse COCO-style JSON annotation
   * Join with metadata CSV (camera ID, location, etc.)
   * Produce `(image, metadata, label)` tuples

2. Modify the config:

   * `num_classes` ≈ 21
   * new label list file
   * different train/val splits

3. Keep the existing:

   * multimodal architecture
   * focal loss
   * two-phase training
   * eval scripts (minor updates)

The transition is straightforward because your pipeline already supports multimodal learning.

---

# **Summary**

You now have:

✓ COCO → TFDS → TF Dataset pipeline
✓ Multilabel model with focal loss
✓ Backbone fine-tuning
✓ Synthetic metadata pipeline
✓ Multimodal architecture (image + metadata)
✓ Evaluation & single-image prediction scripts
✓ Full GPU-accelerated TF 2.10 setup on Windows

You are ready to:

➡ **Switch to Caltech Camera Traps**
➡ **Introduce real metadata**
➡ **Build a high-quality multimodal detection/classification system**

---

If you'd like, I can now help you:

* Start a new CCT loader (`make_cct_dataset()`)
* Write a new `configs/cct_config.json`
* Convert your multimodal scripts for the CCT dataset
* Build a metadata schema (camera ID, time, location, etc.)
* Or help you with clean project restructuring before switching datasets
