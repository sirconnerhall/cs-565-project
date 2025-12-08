# **Object Detection with Metadata - Caltech Camera Traps**

This project implements **object detection models** for the **Caltech Camera Traps (CCT) dataset** with support for **multimodal learning** (image + metadata). The project compares four different detector architectures to evaluate the impact of metadata and two-stage detection approaches.

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Model Architectures](#model-architectures)
3. [Project Structure](#project-structure)
4. [Environment Setup](#environment-setup)
5. [Dataset Setup](#dataset-setup)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Model Comparison](#model-comparison)
9. [Configuration](#configuration)
10. [Known Limitations](#known-limitations)

---

## **Project Overview**

This project implements and compares four object detection models:

- **SSNM**: Single Stage, No Metadata
- **SSM**: Single Stage, with Metadata  
- **TSNM**: Two Stage, No Metadata
- **TSM**: Two Stage, with Metadata

All models use:
- **Anchor-based detection** (YOLO-style)
- **EfficientNet-B0** backbone (transfer learning)
- **Focal loss** for object detection
- **Caltech Camera Traps (CCT) dataset** with real metadata

The goal is to evaluate how **metadata** (location, time, etc.) and **two-stage transfer learning** affect detection performance on camera trap images.

---

## **Model Architectures**

### **Single Stage Models (SSNM, SSM)**
- Direct prediction from backbone features
- Anchor-based detection at a single scale
- Backbone remains frozen during training (single-stage transfer learning)
- SSM includes metadata fusion via FiLM (Feature-wise Linear Modulation) layers

### **Two Stage Models (TSNM, TSM)**
- First stage: objectness prediction
- Second stage: class and bbox refinement
- TSM includes metadata fusion in both stages

### **Metadata Encoding**
Metadata is encoded as an 8-dimensional vector:
- **Location** (1 feature)
- **Hour** (2 features: sin, cos - cyclical encoding)
- **Day of week** (2 features: sin, cos - cyclical encoding)
- **Month** (2 features: sin, cos - cyclical encoding)
- **Brightness** (1 feature: mean pixel intensity)

Cyclical encoding preserves the periodic nature of temporal features (e.g., hour 23:00 is close to hour 00:00).

---

## **Project Structure**

```
basic_image_detection/
├── configs/
│   └── config.json              # Main configuration file
├── models/                      # Models are stored here after training
├── model_comparison_results.json # Results from model comparison
├── src/
│   ├── compare_models.py        # Compare all four models
│   ├── train_all_models.py      # Train all models in sequence
│   ├── SSNM/                    # Single Stage, No Metadata
│   │   ├── build_detector.py
│   │   ├── train_detector.py
│   │   └── eval_detector.py
│   ├── SSM/                     # Single Stage, with Metadata
│   │   ├── build_detector.py
│   │   ├── train_detector.py
│   │   └── eval_detector.py
│   ├── TSNM/                    # Two Stage, No Metadata
│   │   ├── build_detector.py
│   │   └── train_detector.py
│   ├── TSM/                     # Two Stage, with Metadata
│   │   ├── build_detector.py
│   │   ├── train_detector.py
│   │   └── eval_detector.py
│   ├── pipelines/
│   │   ├── cct_tfrecords_pipeline.py  # CCT TFRecords dataset loader
│   │   ├── cct_pipeline.py            # CCT dataset utilities
│   │   ├── cct_splits_utils.py        # CCT split management
│   │   └── convert_cct_to_tfrecords.py # Convert CCT to TFRecords
│   ├── utils/
│   │   ├── detection_utils.py         # Detection loss, metrics, utilities
│   │   ├── anchor_utils.py            # Anchor generation
│   │   ├── anchor_encoder.py          # Anchor encoding/decoding
│   │   ├── metadata_encoding.py       # Metadata encoding utilities
│   │   ├── film_layer.py              # FiLM layer for metadata fusion
│   │   ├── yolo_backbone.py           # Backbone builders (EfficientNet, CSPDarkNet)
│   │   └── load_pretrained_detector.py # Pretrained model loading
│   └── misc_scripts/
│       ├── predict_single_image.py     # Single image prediction
│       └── pick_5_percent_cct.py       # Dataset sampling utilities
└── requirements.txt
```

---

## **Environment Setup**

This project runs on **TensorFlow 2.10.1** on **Windows**.

TF 2.10.1 is the *last* version that supports GPU on Windows.

### **Python Version**
Use **Python 3.10.x**

### **Install CUDA & cuDNN**
TF 2.10.1 requires:
- **CUDA 11.2 – 11.8**
- **cuDNN 8.x**

After installation, ensure:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
```
is on your **PATH**.

### **Create Virtual Environment**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### **GPU Test**
```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```

Expected output:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## **Dataset Setup**

### **Caltech Camera Traps (CCT) Dataset**

The project uses the **Caltech Camera Traps (CCT) dataset**, which includes:
- 21 species classes + empty
- ~66k bounding box annotations
- Images grouped by camera location
- Rich metadata: camera ID, location, timestamp, etc.
- Designed for research on generalization across locations

### **Dataset Paths**

Configure dataset paths in `configs/config.json`:
```json
{
  "cct_images_root": "",
  "cct_annotations": "",
  "cct_bb_annotations": "",
  "cct_splits": "",
  "cct_tfrecords_dir": ""
}
```

### **TFRecords Format**

The project uses **TFRecords** for efficient data loading. To convert CCT dataset to TFRecords:

```bash
python -m src.pipelines.convert_cct_to_tfrecords
```

This will create TFRecord files in the specified `cct_tfrecords_dir` directory.

---

## **Training**

### **Train a Single Model**

Train a specific model:

```bash
# Single Stage, No Metadata
python -m src.SSNM.train_detector

# Single Stage, with Metadata
python -m src.SSM.train_detector

# Two Stage, No Metadata
python -m src.TSNM.train_detector

# Two Stage, with Metadata
python -m src.TSM.train_detector
```

### **Train All Models**

Train all four models in sequence:

```bash
python -m src.train_all_models
```

This will train all models one after another and save them to the `models/` directory.

### **Training Configuration**

All training parameters are configured in `configs/config.json`:
- Image size, batch size, epochs
- Learning rate and fine-tuning settings
- Focal loss parameters
- Anchor configuration
- Dataset paths

### **Model Checkpoints**

Each model saves two checkpoints:
- `{model_name}_best.keras` - Best validation performance
- `{model_name}_last.keras` - Last epoch checkpoint

---

## **Evaluation**

### **Evaluate a Single Model**

Evaluate a specific model:

```bash
# Single Stage, No Metadata
python -m src.SSNM.eval_detector

# Single Stage, with Metadata
python -m src.SSM.eval_detector

# Two Stage, No Metadata
python -m src.TSNM.eval_detector

# Two Stage, with Metadata
python -m src.TSM.eval_detector
```

Evaluation computes:
- **mAP** (mean Average Precision)
- **Precision/Recall**
- **IoU** (Intersection over Union)
- **Class-level metrics**

---

## **Model Comparison**

Compare all four models side-by-side:

```bash
python -m src.compare_models
```

This script:
- Loads all four trained models
- Evaluates them on the validation set
- Computes comprehensive metrics
- Saves results to `model_comparison_results.json`

The comparison includes:
- Detection metrics (mAP, precision, recall)
- Class-level performance
- Average predictions per image
- IoU statistics

---

## **Configuration**

Main configuration file: `configs/config.json`

### **Key Configuration Options**

```json
{
  "image_size": [224, 224],
  "batch_size": 16,
  "epochs": 20,
  "learning_rate": 0.001,
  
  "dataset": "cct",
  "cct_use_tfrecords": true,
  
  "pretrained_model_type": "efficientnet_b0",
  "num_anchors": 3,
  "use_anchors": true,
  
  "focal_gamma": 2.0,
  "focal_alpha": 0.5,
  "bbox_loss_weight": 5.0,
  
  "fine_tune_backbone": true,
  "fine_tune_after_epochs": 5,
  "fine_tune_lr": 5e-5
}
```

## **Known Limitations**

### **1. Model Performance**
Current models show low mAP scores, indicating the detection task is challenging. This may be due to:
- Small object sizes in camera trap images
- Anchor configuration may need tuning
- Loss function weights may need adjustment

### **2. Metadata Impact**
The impact of metadata on detection performance is still being evaluated. Initial results suggest metadata may help with location-specific generalization.

### **3. Two-Stage vs Single-Stage**
The two-stage approach (TSNM, TSM) may require more careful tuning of the objectness threshold and loss weights.

---


## **Summary**

This project provides:
- Four object detection models (SSNM, SSM, TSNM, TSM)
- Caltech Camera Traps dataset support with TFRecords
- Real metadata integration with cyclical encoding
- Anchor-based detection (YOLO-style)
- EfficientNet-B0 and CSPDarkNet backbone support
- Focal loss for object detection
- Training and evaluation scripts for each model
- Model comparison framework
- Full GPU-accelerated TF 2.10 setup on Windows

---