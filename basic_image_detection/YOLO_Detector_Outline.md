# YOLO Detector System Outline

This document provides a comprehensive outline of the YOLO-style detection system, covering the training, evaluation, and utility modules.

---

## Table of Contents

1. [Overview](#overview)
2. [detection_utils.py](#detection_utilspy)
3. [train_yolo_detector.py](#train_yolo_detectorpy)
4. [eval_yolo_detector.py](#eval_yolo_detectorpy)
5. [System Flow](#system-flow)

---

## Overview

The YOLO detector system implements a YOLO-style object detection model with metadata integration for the CCT (Camera Capture Time) dataset. The system consists of three main components:

- **detection_utils.py**: Core utilities for detection operations (IoU, NMS, decoding, mAP)
- **train_yolo_detector.py**: Training pipeline with two-stage fine-tuning
- **eval_yolo_detector.py**: Evaluation and visualization pipeline

---

## detection_utils.py

**Purpose**: Provides essential utilities for object detection operations including box format conversion, non-maximum suppression, prediction decoding, and evaluation metrics.

### Key Functions

#### 1. `compute_iou(box1, box2)`
- **Purpose**: Computes Intersection over Union (IoU) between two bounding boxes
- **Input Format**: Boxes in `[ymin, xmin, ymax, xmax]` format (normalized 0-1)
- **Returns**: IoU value (float)
- **Use Case**: Used for matching predictions to ground truth and NMS

#### 2. `nms(boxes, scores, iou_threshold=0.5, max_output_size=50)`
- **Purpose**: Non-maximum suppression to remove overlapping detections
- **Input**: 
  - Boxes: List of dicts with "bbox" key or list of coordinate arrays
  - Scores: List of scores or extracted from box dicts
- **Parameters**:
  - `iou_threshold`: IoU threshold for suppression (default: 0.5)
  - `max_output_size`: Maximum boxes to keep (default: 50)
- **Returns**: List of indices to keep
- **Algorithm**: Greedy selection of highest-scoring boxes, removing overlapping ones

#### 3. `decode_predictions_grid(grid_pred, num_classes, threshold=0.5, nms_iou=0.5, max_boxes=20, min_box_size=0.01)`
- **Purpose**: Decodes grid-based predictions to bounding box format
- **Input Format**: 
  - `[H, W, 1 + 4 + num_classes]` where:
    - `[..., 0]`: objectness score
    - `[..., 1:5]`: bbox in center format (cx, cy, w, h) normalized
    - `[..., 5:]`: class probabilities
- **Output Format**: List of dicts with keys:
  - `"bbox"`: `[ymin, xmin, ymax, xmax]` (normalized)
  - `"class_id"`: Integer class ID
  - `"score"`: Combined objectness × class confidence
- **Processing Steps**:
  1. Iterate through grid cells
  2. Filter by objectness threshold
  3. Convert center format to corner format
  4. Clip to [0, 1] bounds
  5. Filter by minimum box size
  6. Apply NMS
- **Returns**: List of detection dictionaries

#### 4. `compute_map(predictions_list, ground_truth_list, num_classes, iou_threshold=0.5)`
- **Purpose**: Computes mean Average Precision (mAP) at a given IoU threshold
- **Input**:
  - `predictions_list`: List of lists, each containing prediction dicts per image
  - `ground_truth_list`: List of lists, each containing GT dicts per image
  - `num_classes`: Number of classes
  - `iou_threshold`: IoU threshold for matching (default: 0.5)
- **Algorithm**:
  1. Compute per-class Average Precision (AP)
  2. Match predictions to GT using IoU threshold
  3. Sort predictions by score
  4. Compute precision-recall curve
  5. Use 11-point interpolation for AP
  6. Average AP across classes for mAP
- **Returns**: mAP value (float)

#### 5. `convert_bbox_format(bbox, from_format="cxcywh", to_format="xyxy")`
- **Purpose**: Converts bounding boxes between center format and corner format
- **Formats**:
  - `"cxcywh"`: Center x, center y, width, height
  - `"xyxy"`: xmin, ymin, xmax, ymax
- **Returns**: Converted bbox array

---

## train_yolo_detector.py

**Purpose**: Training pipeline for YOLO-style detection model with metadata integration, implementing a two-stage fine-tuning strategy.

### Main Components

#### Configuration Loading
- **Function**: `load_config(config_name="coco_multilabel_config.json")`
- Loads configuration from JSON file
- Returns config dict and project root path

#### Grid Encoder
- **Function**: `make_grid_encoder_for_detector(num_classes, image_size, grid_size=None)`
- **Purpose**: Converts CCT format annotations to grid format for YOLO training
- **Input Format**: 
  - Images: `[B, H, W, 3]`
  - Targets: Dict with `"bboxes"` `[B, N, 4]` and `"labels"` `[B, N]`
- **Output Format**: `[B, grid_h, grid_w, 1 + 4 + num_classes]`
- **Grid Encoding Logic**:
  1. Converts bboxes from `[ymin, xmin, ymax, xmax]` to center format `(cx, cy, w, h)`
  2. Maps each object to its corresponding grid cell based on center coordinates
  3. Handles multiple objects per cell (keeps largest)
  4. Sets objectness = 1.0, bbox coordinates, and one-hot class encoding
- **Grid Size Estimation**: 
  - Default: Estimates based on image size and backbone downsampling (~32x)
  - Can be explicitly provided

#### Main Training Function

##### Stage 1: Frozen Backbone Training
1. **Model Building**:
   - Uses `build_ssd_detector_with_metadata()` from `build_yolo_detector.py`
   - Integrates metadata (location, time, brightness) with image features
   - Backbone starts frozen

2. **Dataset Preparation**:
   - Supports both TFRecords and JSON pipeline
   - Loads CCT dataset with metadata extraction
   - Combines image metadata with computed brightness
   - Applies grid encoding to targets

3. **Loss and Metrics**:
   - `DetectionLossFocal`: Focal loss for detection
   - Component loss metrics for monitoring
   - `objectness_accuracy`: Accuracy metric

4. **Training Configuration**:
   - Optimizer: Adam with configurable learning rate
   - Callbacks:
     - ModelCheckpoint (saves best model)
     - EarlyStopping (patience=10)
     - ReduceLROnPlateau
     - PredictionStats (custom callback for detection stats)

##### Stage 2: Fine-tuning with Unfrozen Backbone
1. **Backbone Unfreezing**:
   - Identifies backbone layers (MobileNet, ResNet, EfficientNet)
   - Unfreezes all backbone layers
   - Lowers learning rate (default: 0.1 × Stage 1 LR)

2. **Continued Training**:
   - Uses same loss and metrics
   - Continues from Stage 1 epoch
   - More aggressive early stopping (patience=5)
   - Restores best weights on early stop

### Key Features

- **Metadata Integration**: Combines CCT metadata (location_id, hour, day_of_week, month) with image brightness
- **Two-Stage Training**: Freezes backbone initially, then fine-tunes
- **Flexible Dataset Loading**: Supports TFRecords (faster) or JSON pipeline
- **Grid-Based Encoding**: Converts variable-length annotations to fixed-size grid format
- **Model Checkpointing**: Saves best and last models

---

## eval_yolo_detector.py

**Purpose**: Evaluation and visualization pipeline for trained YOLO detection models.

### Main Components

#### Configuration Loading
- **Function**: `load_config()`
- Loads configuration and project root
- Determines model path from config

#### Visualization
- **Function**: `draw_boxes(image, gt_boxes, pred_boxes, class_names)`
- **Purpose**: Visualizes ground truth and predicted boxes on images
- **Visualization Details**:
  - Ground truth: Green boxes with "GT: {class_name}" labels
  - Predictions: Red boxes with "{class_name} ({score})" labels
  - Uses matplotlib patches for drawing

#### Main Evaluation Function

##### 1. Model Loading
- Loads trained model from checkpoint
- Handles custom objects (loss functions, metrics)
- Supports models trained with `DetectionLossFocal` and component metrics

##### 2. Dataset Preparation
- Loads validation set from CCT dataset
- Supports TFRecords or JSON pipeline
- Adds metadata (same format as training)
- For TFRecords: Uses placeholder metadata if not available

##### 3. Prediction Collection
- Iterates through validation batches
- Runs model inference (with metadata if available)
- Decodes predictions using `decode_predictions_grid()`
- Collects:
  - Predictions: List of detection dicts per image
  - Ground truth: List of GT dicts per image
  - Images: For visualization

##### 4. Evaluation Metrics
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.75**: Mean Average Precision at IoU threshold 0.75
- Uses `compute_map()` from `detection_utils.py`

##### 5. Results Display
- Prints evaluation metrics
- Shows example predictions (first 5 images)
- Displays counts of GT vs predicted boxes
- Visualizes images with predictions (first 5 with detections)

### Evaluation Process Flow

1. Load model and configuration
2. Prepare validation dataset with metadata
3. Run inference on batches (limited to first 10 batches for speed)
4. Decode grid predictions to bounding boxes
5. Extract ground truth boxes from targets
6. Compute mAP metrics
7. Visualize results

### Key Features

- **Comprehensive Metrics**: Computes mAP at multiple IoU thresholds
- **Visualization**: Draws GT and predictions on images
- **Flexible Input**: Handles both metadata-enabled and metadata-free models
- **Efficient Evaluation**: Limits batch processing for quick results
- **Detailed Output**: Shows per-image statistics and example predictions

---

## System Flow

### Training Flow
```
1. Load Configuration
   ↓
2. Load CCT Dataset (train/val splits)
   ↓
3. Extract Metadata Features
   ↓
4. Build Model (with frozen backbone)
   ↓
5. Encode Targets to Grid Format
   ↓
6. Stage 1: Train with Frozen Backbone
   ↓
7. Unfreeze Backbone
   ↓
8. Stage 2: Fine-tune with Unfrozen Backbone
   ↓
9. Save Best and Last Models
```

### Evaluation Flow
```
1. Load Configuration and Trained Model
   ↓
2. Load Validation Dataset
   ↓
3. Add Metadata
   ↓
4. Run Inference (Batch Processing)
   ↓
5. Decode Grid Predictions
   ↓
6. Extract Ground Truth
   ↓
7. Compute mAP Metrics
   ↓
8. Visualize Results
```

### Data Format Conversions

**Training Input**:
- Images: `[B, H, W, 3]` (normalized 0-1)
- Metadata: `[B, 5]` (location_id, hour, day_of_week, month, brightness)
- Targets: Dict with `"bboxes"` `[B, N, 4]` and `"labels"` `[B, N]`

**Grid Encoding**:
- Targets → `[B, grid_h, grid_w, 1 + 4 + num_classes]`
- Format: `[objectness, cx, cy, w, h, class_one_hot]`

**Model Output**:
- Predictions: `[B, grid_h, grid_w, 1 + 4 + num_classes]`

**Decoded Output**:
- List of dicts: `[{"bbox": [ymin, xmin, ymax, xmax], "class_id": int, "score": float}, ...]`

---

## Dependencies

### External Libraries
- `tensorflow` / `keras`: Deep learning framework
- `numpy`: Numerical operations
- `matplotlib`: Visualization

### Internal Modules
- `cct_pipeline`: CCT dataset loading and metadata extraction
- `cct_tfrecords_pipeline`: TFRecords dataset loading
- `build_yolo_detector`: Model architecture
- `train_cct_multimodal_detector`: Loss functions and metrics
- `cct_splits_utils`: Dataset split management

---

## Configuration Parameters

Key configuration parameters (from `coco_multilabel_config.json`):

- `image_size`: Input image dimensions
- `batch_size`: Training batch size
- `epochs`: Total training epochs
- `learning_rate`: Initial learning rate
- `freeze_backbone_epochs`: Epochs to train with frozen backbone
- `fine_tune_lr`: Learning rate for Stage 2 (default: 0.1 × learning_rate)
- `focal_gamma`: Focal loss gamma parameter
- `focal_alpha`: Focal loss alpha parameter
- `positive_weight`: Weight for positive examples
- `detector_model_name`: Model name for saving
- `pretrained_model_type`: Backbone type (e.g., "ssd_mobilenet_v2")
- `cct_use_tfrecords`: Whether to use TFRecords format
- `cct_tfrecords_dir`: Path to TFRecords directory

---

## Notes

- The system is designed specifically for the CCT dataset but can be adapted
- Grid size is automatically determined from model output shape
- Metadata integration is optional (uses placeholders if not available)
- Evaluation is limited to first 10 batches for speed (configurable)
- Model supports both single-modal (image only) and multi-modal (image + metadata) inference

