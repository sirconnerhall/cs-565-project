# Multi-Stage Transfer Learning with Metadata Integration for Wildlife Image Classification
## Complete Implementation Guide: Zero to Hero

**Project:** CS 465/565 Wildlife Image Classification  
**Authors:** Conner Hall and Will Morrow  
**Methodology:** Multi-stage transfer learning with spatial and temporal metadata integration

---

## Table of Contents
1. [Overview and Methodology](#overview-and-methodology)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Architecture Implementation](#architecture-implementation)
5. [Multi-Stage Training Pipeline](#multi-stage-training-pipeline)
6. [Evaluation and Analysis](#evaluation-and-analysis)
7. [Results Generation for Paper](#results-generation-for-paper)
8. [Troubleshooting](#troubleshooting)

---

## Overview and Methodology

### Core Approach
This project implements a **hierarchical transfer learning pipeline** that:
1. Starts with general visual features (ImageNet pretrained)
2. Progresses through wildlife-specific representations (intermediate wildlife dataset)
3. Specializes to the target species set (Snapshot Serengeti or similar)

### Key Innovation
**At each stage**, we fuse image features with spatial and temporal metadata including:
- **Temporal information:** date, time (encoded cyclically with sine-cosine functions)
- **Spatial data:** camera location (latitude/longitude)

This allows the model to leverage:
- Species activity cycles (diurnal/nocturnal patterns)
- Seasonal presence
- Habitat preferences
- Geographic distribution patterns

### Three Main Contributions
1. **Multi-stage transfer learning pipeline** for wildlife conservation with progressive domain specialization
2. **Metadata integration throughout the pipeline** (not just at the end)
3. **Evidence that combining both techniques** outperforms either approach individually

### Architecture Components (Based on Literature Review)

#### From Liu et al. (2024) - Temporal-SE-ResNet50:
- **Base:** ResNet50 architecture
- **SE blocks:** Squeeze-and-excitation attention mechanism for emphasizing informative visual features
- **Temporal encoding:** Sine-cosine cyclical encoding for date/time preservation
- **Metadata MLP:** Residual multilayer perceptron for temporal feature transformation
- **Fusion module:** Dynamic MLP for adaptive reweighting of image and metadata features

#### From Zhang et al. (2024) - Multi-stage Framework:
- **Stage 1:** ImageNet pretraining (generic visual representations)
- **Stage 2:** Intermediate wildlife dataset adaptation (wildlife-specific features)
- **Stage 3:** Target dataset fine-tuning (task-specific specialization)
- **Sequential knowledge transfer** with hierarchical feature learning

---

## Environment Setup

### Step 1: Create Python Environment

```bash
# Create a new virtual environment
python3 -m venv wildlife_classifier_env

# Activate the environment
# On Linux/Mac:
source wildlife_classifier_env/bin/activate
# On Windows:
wildlife_classifier_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install PyTorch (check https://pytorch.org for your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision torchaudio

# Install other required packages
pip install pandas numpy scikit-learn matplotlib seaborn
pip install pillow opencv-python tqdm jupyter
pip install tensorboard  # For training visualization
```

### Step 3: Create Project Directory Structure

```bash
# Create the project structure
mkdir -p wildlife_classifier/{data/{raw,processed,stage1,stage2,stage3},models,results,src,notebooks}

cd wildlife_classifier

# Data subdirectories
mkdir -p data/stage1/{images,splits}
mkdir -p data/stage2/{images,splits}
mkdir -p data/stage3/{images,splits}

# Results subdirectories
mkdir -p results/{figures,tables,checkpoints}
```

Your final structure should look like:
```
wildlife_classifier/
├── data/
│   ├── raw/                    # Original downloaded data
│   ├── processed/              # Preprocessed images
│   ├── stage1/                 # ImageNet (already pretrained - skip)
│   ├── stage2/                 # General wildlife dataset
│   │   ├── images/
│   │   ├── splits/
│   │   └── metadata.csv
│   └── stage3/                 # Target dataset (Snapshot Serengeti)
│       ├── images/
│       ├── splits/
│       └── metadata.csv
├── models/                     # Saved model checkpoints
├── results/                    # Figures, tables, analysis
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
├── src/                        # Source code
├── notebooks/                  # Jupyter notebooks for analysis
└── README.md
```

---

## Data Preparation

### Step 4: Obtain Datasets

#### Stage 2: General Wildlife Dataset
Options:
- **iNaturalist** (subset of mammal classes)
- **COCO Animals** (subset)
- **Caltech Camera Traps**

For this project, we'll use a subset of wildlife images that represents broader wildlife categories.

#### Stage 3: Target Dataset
- **Snapshot Serengeti** (recommended) or similar camera trap dataset
- **DrivenData competition dataset** (if available)

### Step 5: Create Metadata CSV Files

Your metadata files should have this structure:

**`metadata.csv` format:**
```csv
image_id,species,datetime,latitude,longitude,camera_id
img_0001.jpg,lion,2023-06-15 14:30:00,-2.3333,34.8333,CAM_001
img_0002.jpg,elephant,2023-06-15 06:15:00,-2.3340,34.8340,CAM_002
img_0003.jpg,zebra,2023-06-16 18:45:00,-2.3350,34.8350,CAM_003
...
```

**Required columns:**
- `image_id`: Filename of the image
- `species`: Target class label (species name)
- `datetime`: Timestamp in format `YYYY-MM-DD HH:MM:SS`
- `latitude`: Camera latitude coordinate
- `longitude`: Camera longitude coordinate
- `camera_id`: Unique identifier for each camera trap

### Step 6: Data Preprocessing Script

Create `src/data_preprocessing.py`:

```python
"""
Data preprocessing and loading for multi-stage wildlife classification.
Implements cyclical temporal encoding and spatial feature extraction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from datetime import datetime

class WildlifeDataset(Dataset):
    """
    Wildlife camera trap dataset with metadata integration.
    
    Implements cyclical encoding for temporal features and spatial normalization
    as described in Liu et al. (2024).
    """
    
    def __init__(self, image_dir, metadata_path, transform=None, 
                 include_metadata=True, split='train'):
        """
        Args:
            image_dir (str): Directory containing images
            metadata_path (str): Path to metadata CSV file
            transform: Torchvision transforms to apply
            include_metadata (bool): Whether to include metadata features
            split (str): 'train', 'val', or 'test'
        """
        self.image_dir = image_dir
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform
        self.include_metadata = include_metadata
        self.split = split
        
        # Create species label encoding
        self.species_to_idx = {species: idx for idx, species in 
                               enumerate(sorted(self.metadata['species'].unique()))}
        self.idx_to_species = {idx: species for species, idx in 
                               self.species_to_idx.items()}
        self.num_classes = len(self.species_to_idx)
        
        print(f"Loaded {len(self.metadata)} images with {self.num_classes} species")
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Returns:
            If include_metadata=True: (image, metadata_features, label)
            If include_metadata=False: (image, label)
        """
        # Load image
        img_name = self.metadata.iloc[idx]['image_id']
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        species = self.metadata.iloc[idx]['species']
        label = self.species_to_idx[species]
        
        if self.include_metadata:
            # Extract and encode metadata features
            metadata_features = self.extract_metadata(idx)
            return image, metadata_features, label
        else:
            return image, label
    
    def extract_metadata(self, idx):
        """
        Extract and encode temporal and spatial metadata.
        
        Implements cyclical encoding using sine-cosine functions for temporal features
        as described in Liu et al. (2024) to preserve periodic nature.
        """
        row = self.metadata.iloc[idx]
        
        # Parse datetime
        datetime_str = row['datetime']
        datetime_obj = pd.to_datetime(datetime_str)
        
        # === TEMPORAL ENCODING (Cyclical with sine-cosine) ===
        
        # Hour of day (0-23) -> cyclical encoding
        # This ensures 23:00 and 00:00 are close in feature space
        hour = datetime_obj.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Day of year (1-365) -> cyclical encoding
        # This ensures Dec 31 and Jan 1 are close in feature space
        day_of_year = datetime_obj.dayofyear
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)
        
        # Month (1-12) -> cyclical encoding
        month = datetime_obj.month
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # === SPATIAL ENCODING ===
        
        # Normalize latitude to [-1, 1] (valid range: -90 to 90)
        lat = row['latitude'] / 90.0
        
        # Normalize longitude to [-1, 1] (valid range: -180 to 180)
        lon = row['longitude'] / 180.0
        
        # Combine all metadata features into single vector
        # Total: 8 features (6 temporal + 2 spatial)
        metadata = np.array([
            hour_sin, hour_cos,      # Hour encoding (2 features)
            day_sin, day_cos,        # Day of year encoding (2 features)
            month_sin, month_cos,    # Month encoding (2 features)
            lat, lon                 # Spatial encoding (2 features)
        ], dtype=np.float32)
        
        return torch.from_numpy(metadata)
    
    def get_class_weights(self):
        """Calculate class weights for handling imbalanced datasets"""
        species_counts = self.metadata['species'].value_counts()
        total = len(self.metadata)
        weights = {self.species_to_idx[species]: total / count 
                   for species, count in species_counts.items()}
        return torch.tensor([weights[i] for i in range(self.num_classes)])


def get_transforms(augment=True):
    """
    Get image transforms for training and validation.
    
    Args:
        augment (bool): Whether to apply data augmentation (for training)
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                   saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(image_dir, metadata_path, batch_size=32, 
                       num_workers=4, include_metadata=True):
    """
    Create train and validation data loaders.
    
    Args:
        image_dir (str): Directory containing images
        metadata_path (str): Path to metadata CSV
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        include_metadata (bool): Whether to include metadata features
    
    Returns:
        tuple: (train_loader, val_loader, dataset)
    """
    # Load full dataset
    dataset = WildlifeDataset(
        image_dir=image_dir,
        metadata_path=metadata_path,
        transform=None,  # Will set per split
        include_metadata=include_metadata
    )
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use random split with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Apply appropriate transforms
    train_dataset.dataset.transform = get_transforms(augment=True)
    val_dataset.dataset.transform = get_transforms(augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, dataset


if __name__ == '__main__':
    # Test the dataset
    print("Testing dataset loading...")
    
    # Example usage
    dataset = WildlifeDataset(
        image_dir='../data/stage3/images',
        metadata_path='../data/stage3/metadata.csv',
        transform=get_transforms(augment=False),
        include_metadata=True
    )
    
    # Get a sample
    image, metadata, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Metadata values: {metadata}")
    print(f"Label: {label} ({dataset.idx_to_species[label]})")
    print(f"Number of classes: {dataset.num_classes}")
```

---

## Architecture Implementation

### Step 7: Implement Model Architecture

Create `src/model_architecture.py`:

```python
"""
Model architecture for multi-stage transfer learning with metadata integration.

Implements:
1. Temporal-SE-ResNet50 (Liu et al. 2024) for visual feature extraction
2. Metadata encoder with residual MLP
3. Dynamic fusion module for adaptive feature combination
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.
    
    From Liu et al. (2024): "augmented with a squeeze-and-excitation (SE) 
    attention mechanism that allows the network to emphasize informative 
    visual features and suppress irrelevant background noise."
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch, channels, _, _ = x.size()
        # Squeeze: Global information embedding
        y = self.squeeze(x).view(batch, channels)
        # Excitation: Adaptive recalibration
        y = self.excitation(y).view(batch, channels, 1, 1)
        # Scale original features
        return x * y.expand_as(x)


class MetadataEncoder(nn.Module):
    """
    Residual MLP for encoding temporal and spatial metadata.
    
    From Liu et al. (2024): "passed through a residual multilayer perceptron 
    (MLP) network that transforms them into a feature representation reflecting 
    seasonal and daily rhythms of animal activity."
    """
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=128, dropout=0.3):
        super(MetadataEncoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection from input to first hidden layer
        self.residual = nn.Linear(input_dim, hidden_dim)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: Metadata features [batch_size, 8]
                (hour_sin, hour_cos, day_sin, day_cos, 
                 month_sin, month_cos, lat, lon)
        Returns:
            Encoded metadata [batch_size, output_dim]
        """
        # First layer with residual connection
        identity = self.residual(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Add residual
        out = out + identity
        
        # Second layer
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Final projection
        out = self.fc3(out)
        
        return out


class DynamicFusionModule(nn.Module):
    """
    Dynamic MLP module for adaptive fusion of image and metadata features.
    
    From Liu et al. (2024): "merged using a dynamic MLP module that adaptively 
    reweights and refines the combined representation through learned projections, 
    enhancing the discriminative power of the final model."
    """
    def __init__(self, image_dim=2048, metadata_dim=128, output_dim=512, dropout=0.3):
        super(DynamicFusionModule, self).__init__()
        
        # Project both features to common dimension
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for adaptive weighting
        # Learns to balance image vs metadata importance
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, image_features, metadata_features):
        """
        Args:
            image_features: Visual features [batch_size, image_dim]
            metadata_features: Encoded metadata [batch_size, metadata_dim]
        Returns:
            Fused features [batch_size, output_dim]
        """
        # Project to common dimension
        img_proj = self.image_proj(image_features)
        meta_proj = self.metadata_proj(metadata_features)
        
        # Compute attention weights
        concat = torch.cat([img_proj, meta_proj], dim=1)
        weights = self.attention(concat)  # [batch_size, 2]
        
        # Apply weighted fusion
        fused = weights[:, 0:1] * img_proj + weights[:, 1:2] * meta_proj
        
        # Final transformation
        out = self.fusion(fused)
        
        return out


class MultiStageWildlifeClassifier(nn.Module):
    """
    Complete multi-stage wildlife classifier with metadata integration.
    
    Architecture:
    1. ResNet50 backbone with SE blocks (visual feature extraction)
    2. Metadata encoder (temporal/spatial feature encoding)
    3. Dynamic fusion module (adaptive feature combination)
    4. Classification head
    
    Implements methodology from:
    - Liu et al. (2024): Temporal-SE-ResNet50 architecture
    - Zhang et al. (2024): Multi-stage transfer learning framework
    """
    def __init__(self, num_classes, pretrained=True, metadata_dim=8, 
                 use_se_blocks=True, fusion_dim=512):
        super(MultiStageWildlifeClassifier, self).__init__()
        
        # Load ResNet50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get number of features from backbone
        num_features = self.backbone.fc.in_features  # 2048 for ResNet50
        
        # Replace final FC layer with identity (we'll add our own classifier)
        self.backbone.fc = nn.Identity()
        
        # Optionally add SE blocks to ResNet layers
        if use_se_blocks:
            self._add_se_blocks()
        
        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(
            input_dim=metadata_dim,
            hidden_dim=64,
            output_dim=128,
            dropout=0.3
        )
        
        # Dynamic fusion module
        self.fusion = DynamicFusionModule(
            image_dim=num_features,
            metadata_dim=128,
            output_dim=fusion_dim,
            dropout=0.3
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def _add_se_blocks(self):
        """Add SE blocks to ResNet layers for channel attention."""
        # Add SE block after each residual block in layer4 (most important)
        for name, module in self.backbone.layer4.named_children():
            if isinstance(module, models.resnet.Bottleneck):
                # Insert SE block before final relu
                channels = module.conv3.out_channels
                module.se = SEBlock(channels, reduction=16)
                # Modify forward to include SE
                original_forward = module.forward
                def new_forward(x, original_forward=original_forward, se=module.se):
                    identity = x
                    out = module.conv1(x)
                    out = module.bn1(out)
                    out = module.relu(out)
                    out = module.conv2(out)
                    out = module.bn2(out)
                    out = module.relu(out)
                    out = module.conv3(out)
                    out = module.bn3(out)
                    out = se(out)  # Apply SE block
                    if module.downsample is not None:
                        identity = module.downsample(x)
                    out += identity
                    out = module.relu(out)
                    return out
                module.forward = new_forward
    
    def forward(self, images, metadata=None):
        """
        Forward pass with optional metadata.
        
        Args:
            images: Input images [batch_size, 3, 224, 224]
            metadata: Optional metadata features [batch_size, 8]
        
        Returns:
            Logits [batch_size, num_classes]
        """
        # Extract visual features from backbone
        image_features = self.backbone(images)  # [batch_size, 2048]
        
        if metadata is not None:
            # Encode metadata
            metadata_features = self.metadata_encoder(metadata)  # [batch_size, 128]
            
            # Fuse image and metadata features
            fused_features = self.fusion(image_features, metadata_features)
        else:
            # Image-only mode (for baseline comparisons)
            # Project image features directly
            fused_features = self.fusion.image_proj(image_features)
        
        # Classify
        output = self.classifier(fused_features)
        
        return output
    
    def freeze_backbone(self, freeze=True):
        """
        Freeze or unfreeze the entire backbone.
        Used during transfer learning stages.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def unfreeze_last_n_layers(self, n=2):
        """
        Unfreeze only the last n layers of the backbone.
        Useful for gradual fine-tuning.
        
        Args:
            n: Number of layer groups to unfreeze (1-4 for ResNet)
        """
        # Freeze all first
        self.freeze_backbone(True)
        
        # Get layer names in reverse order
        layers = ['layer4', 'layer3', 'layer2', 'layer1']
        
        # Unfreeze last n layers
        for layer_name in layers[:n]:
            layer = getattr(self.backbone, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self):
        """Get count of total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(num_classes, pretrained=True, metadata_dim=8):
    """
    Factory function to create the model.
    
    Args:
        num_classes: Number of species classes
        pretrained: Whether to use ImageNet pretrained weights
        metadata_dim: Dimension of metadata features (default: 8)
    
    Returns:
        MultiStageWildlifeClassifier instance
    """
    model = MultiStageWildlifeClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        metadata_dim=metadata_dim,
        use_se_blocks=True,
        fusion_dim=512
    )
    
    print(f"Model created with {model.get_total_parameters():,} total parameters")
    print(f"Trainable parameters: {model.get_trainable_parameters():,}")
    
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing model architecture...")
    
    # Create model
    model = create_model(num_classes=48, pretrained=True)
    
    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    metadata = torch.randn(batch_size, 8)
    
    # Forward pass with metadata
    output = model(images, metadata)
    print(f"Output shape: {output.shape}")  # Should be [4, 48]
    
    # Forward pass without metadata
    output_no_meta = model(images, None)
    print(f"Output shape (no metadata): {output_no_meta.shape}")
    
    # Test freezing
    model.freeze_backbone(True)
    print(f"Trainable parameters (frozen): {model.get_trainable_parameters():,}")
    
    model.unfreeze_last_n_layers(2)
    print(f"Trainable parameters (last 2 layers): {model.get_trainable_parameters():,}")
```

---

## Multi-Stage Training Pipeline

### Step 8: Implement Training Logic

Create `src/training_pipeline.py`:

```python
"""
Multi-stage training pipeline for wildlife classification.

Implements the hierarchical transfer learning framework from Zhang et al. (2024):
Stage 1: ImageNet pretraining (skip - use pretrained weights)
Stage 2: General wildlife dataset
Stage 3: Target-specific dataset

At each stage, metadata is integrated throughout the pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime


class MultiStageTrainer:
    """
    Trainer for multi-stage transfer learning pipeline.
    
    Handles:
    - Stage-specific training configurations
    - Backbone freezing strategies
    - Learning rate scheduling
    - Model checkpointing
    - Training history tracking
    """
    
    def __init__(self, model, device='cuda', log_dir='runs'):
        """
        Args:
            model: MultiStageWildlifeClassifier instance
            device: Device to train on ('cuda' or 'cpu')
            log_dir: Directory for TensorBoard logs
        """
        self.model = model.to(device)
        self.device = device
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # TensorBoard writer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'{log_dir}/experiment_{timestamp}')
        
        print(f"Trainer initialized on device: {device}")
        print(f"Model has {model.get_trainable_parameters():,} trainable parameters")
    
    def train_stage(self, train_loader, val_loader, stage_name, num_epochs=20,
                    learning_rate=0.001, weight_decay=1e-4, freeze_backbone=False,
                    unfreeze_last_n=0, save_dir='models', patience=5):
        """
        Train a single stage of the multi-stage pipeline.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            stage_name: Name of current stage (e.g., 'stage2', 'stage3')
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            freeze_backbone: Whether to freeze the backbone entirely
            unfreeze_last_n: Number of backbone layers to unfreeze (0 = unfreeze all)
            save_dir: Directory to save model checkpoints
            patience: Early stopping patience
        
        Returns:
            best_val_acc: Best validation accuracy achieved
        """
        print(f"\n{'='*70}")
        print(f"Training {stage_name}")
        print(f"{'='*70}")
        
        # Configure backbone freezing based on stage
        if freeze_backbone:
            self.model.freeze_backbone(True)
            print("Status: Backbone frozen, training classifier only")
        elif unfreeze_last_n > 0:
            self.model.unfreeze_last_n_layers(unfreeze_last_n)
            print(f"Status: Last {unfreeze_last_n} backbone layers unfrozen")
        else:
            self.model.freeze_backbone(False)
            print("Status: Full model trainable (fine-tuning)")
        
        print(f"Trainable parameters: {self.model.get_trainable_parameters():,}")
        
        # Setup optimizer - only optimize parameters with requires_grad=True
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - reduce on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training tracking
        best_val_acc = 0.0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        global_step = 0
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # === TRAINING PHASE ===
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc='Training')
            for batch_idx, batch in enumerate(pbar):
                # Unpack batch
                if len(batch) == 3:  # With metadata
                    images, metadata, labels = batch
                    images = images.to(self.device)
                    metadata = metadata.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images, metadata)
                else:  # Without metadata
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images, None)
                
                # Forward pass
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                current_acc = 100.0 * train_correct / train_total
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
                
                # TensorBoard logging
                global_step += 1
                self.writer.add_scalar(f'{stage_name}/train_loss_step', 
                                      loss.item(), global_step)
            
            # Calculate epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100.0 * train_correct / train_total
            
            # === VALIDATION PHASE ===
            val_loss, val_accuracy, val_top5_acc = self.evaluate(
                val_loader, criterion
            )
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_accuracy)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # TensorBoard logging
            self.writer.add_scalar(f'{stage_name}/train_loss_epoch', 
                                  avg_train_loss, epoch)
            self.writer.add_scalar(f'{stage_name}/train_acc', 
                                  train_accuracy, epoch)
            self.writer.add_scalar(f'{stage_name}/val_loss', 
                                  val_loss, epoch)
            self.writer.add_scalar(f'{stage_name}/val_acc', 
                                  val_accuracy, epoch)
            self.writer.add_scalar(f'{stage_name}/val_top5_acc', 
                                  val_top5_acc, epoch)
            self.writer.add_scalar(f'{stage_name}/learning_rate', 
                                  optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Val Top-5: {val_top5_acc:.2f}%")
            
            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save checkpoint
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f'{stage_name}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_accuracy,
                    'val_loss': val_loss,
                    'val_top5_acc': val_top5_acc,
                    'history': self.history
                }, checkpoint_path)
                
                print(f"  ✓ New best model saved! (Val Acc: {val_accuracy:.2f}%)")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
        
        print(f"\n{stage_name} Training Complete!")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        
        return best_val_acc
    
    def evaluate(self, data_loader, criterion):
        """
        Evaluate model on validation or test set.
        
        Args:
            data_loader: Data loader for evaluation
            criterion: Loss function
        
        Returns:
            tuple: (average_loss, accuracy, top5_accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        top5_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch
                if len(batch) == 3:
                    images, metadata, labels = batch
                    images = images.to(self.device)
                    metadata = metadata.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images, metadata)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images, None)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Top-1 accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
                top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        top5_accuracy = 100.0 * top5_correct / total
        
        return avg_loss, accuracy, top5_accuracy
    
    def save_history(self, filepath='training_history.json'):
        """Save training history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {filepath}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Acc: {checkpoint['val_acc']:.2f}%")
        return checkpoint
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


if __name__ == '__main__':
    print("Training pipeline module loaded successfully")
    print("Import this module in your main training script")
```

### Step 9: Main Training Script

Create `src/main_training.py`:

```python
"""
Main training script for multi-stage transfer learning pipeline.

Implements the complete training procedure:
Stage 1: ImageNet pretrained (starting point)
Stage 2: General wildlife dataset with metadata
Stage 3: Target dataset with metadata and full fine-tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse
import os

from model_architecture import create_model
from data_preprocessing import create_data_loaders, WildlifeDataset
from training_pipeline import MultiStageTrainer


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """Main training function."""
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Multi-Stage Transfer Learning for Wildlife Classification")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    
    # ========================================================================
    # STAGE 1: ImageNet Pretrained Weights (Starting Point)
    # ========================================================================
    print(f"\n{'='*70}")
    print("STAGE 1: ImageNet Pretrained Weights")
    print(f"{'='*70}")
    print("Using ResNet50 pretrained on ImageNet as starting point")
    print("This provides general visual feature representations")
    print("Skip explicit training - weights already optimized")
    
    # ========================================================================
    # STAGE 2: General Wildlife Dataset
    # ========================================================================
    if not args.skip_stage2:
        print(f"\n{'='*70}")
        print("STAGE 2: General Wildlife Dataset Training")
        print(f"{'='*70}")
        print("Objective: Learn wildlife-specific visual features")
        print("Dataset: Broader wildlife categories (multiple ecosystems)")
        print("Strategy: Metadata integration with frozen then fine-tuned backbone")
        
        # Load Stage 2 data
        print("\nLoading Stage 2 data...")
        stage2_train_loader, stage2_val_loader, stage2_dataset = create_data_loaders(
            image_dir=args.stage2_image_dir,
            metadata_path=args.stage2_metadata,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            include_metadata=True
        )
        
        num_stage2_classes = stage2_dataset.num_classes
        print(f"Stage 2 classes: {num_stage2_classes}")
        
        # Create model with ImageNet weights
        model = create_model(
            num_classes=num_stage2_classes,
            pretrained=True,
            metadata_dim=8
        )
        
        # Initialize trainer
        trainer = MultiStageTrainer(model, device=device, log_dir=args.log_dir)
        
        # Phase 2a: Train classifier with frozen backbone
        print("\n--- Stage 2a: Training classifier (backbone frozen) ---")
        stage2a_acc = trainer.train_stage(
            train_loader=stage2_train_loader,
            val_loader=stage2_val_loader,
            stage_name='stage2a_frozen',
            num_epochs=args.stage2_epochs_frozen,
            learning_rate=args.stage2_lr_frozen,
            freeze_backbone=True,
            save_dir=args.save_dir,
            patience=args.patience
        )
        
        # Phase 2b: Fine-tune last layers
        print("\n--- Stage 2b: Fine-tuning last 2 layers ---")
        stage2b_acc = trainer.train_stage(
            train_loader=stage2_train_loader,
            val_loader=stage2_val_loader,
            stage_name='stage2b_finetune',
            num_epochs=args.stage2_epochs_finetune,
            learning_rate=args.stage2_lr_finetune,
            unfreeze_last_n=2,
            save_dir=args.save_dir,
            patience=args.patience
        )
        
        print(f"\nStage 2 Complete:")
        print(f"  Phase 2a (frozen) accuracy: {stage2a_acc:.2f}%")
        print(f"  Phase 2b (fine-tuned) accuracy: {stage2b_acc:.2f}%")
        
        # Save training history
        trainer.save_history(os.path.join(args.save_dir, 'stage2_history.json'))
        trainer.close()
    
    # ========================================================================
    # STAGE 3: Target Dataset (Snapshot Serengeti or similar)
    # ========================================================================
    print(f"\n{'='*70}")
    print("STAGE 3: Target Dataset Training")
    print(f"{'='*70}")
    print("Objective: Specialize to target species and ecosystem")
    print("Dataset: Snapshot Serengeti (or target camera trap dataset)")
    print("Strategy: Transfer from Stage 2, integrate metadata, full fine-tuning")
    
    # Load Stage 3 data
    print("\nLoading Stage 3 data...")
    stage3_train_loader, stage3_val_loader, stage3_dataset = create_data_loaders(
        image_dir=args.stage3_image_dir,
        metadata_path=args.stage3_metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_metadata=True
    )
    
    num_stage3_classes = stage3_dataset.num_classes
    print(f"Stage 3 classes: {num_stage3_classes}")
    
    # Create new model with target number of classes
    model = create_model(
        num_classes=num_stage3_classes,
        pretrained=False,  # We'll load Stage 2 weights manually
        metadata_dim=8
    )
    
    # Load Stage 2 weights (if not skipped)
    if not args.skip_stage2:
        stage2_checkpoint_path = os.path.join(args.save_dir, 'stage2b_finetune_best.pth')
        if os.path.exists(stage2_checkpoint_path):
            print(f"\nLoading Stage 2 weights from: {stage2_checkpoint_path}")
            checkpoint = torch.load(stage2_checkpoint_path, map_location=device)
            stage2_state = checkpoint['model_state_dict']
            
            # Transfer weights except final classifier (different number of classes)
            model_state = model.state_dict()
            pretrained_dict = {k: v for k, v in stage2_state.items() 
                             if k in model_state and 'classifier' not in k}
            model_state.update(pretrained_dict)
            model.load_state_dict(model_state)
            print("Stage 2 weights transferred successfully (except classifier)")
        else:
            print(f"Warning: Stage 2 checkpoint not found at {stage2_checkpoint_path}")
            print("Using ImageNet weights only")
    
    # Initialize trainer
    trainer = MultiStageTrainer(model, device=device, log_dir=args.log_dir)
    
    # Phase 3a: Train classifier with frozen backbone
    print("\n--- Stage 3a: Training classifier (backbone frozen) ---")
    stage3a_acc = trainer.train_stage(
        train_loader=stage3_train_loader,
        val_loader=stage3_val_loader,
        stage_name='stage3a_frozen',
        num_epochs=args.stage3_epochs_frozen,
        learning_rate=args.stage3_lr_frozen,
        freeze_backbone=True,
        save_dir=args.save_dir,
        patience=args.patience
    )
    
    # Phase 3b: Fine-tune entire model
    print("\n--- Stage 3b: Full model fine-tuning ---")
    stage3b_acc = trainer.train_stage(
        train_loader=stage3_train_loader,
        val_loader=stage3_val_loader,
        stage_name='stage3b_finetune',
        num_epochs=args.stage3_epochs_finetune,
        learning_rate=args.stage3_lr_finetune,
        freeze_backbone=False,
        save_dir=args.save_dir,
        patience=args.patience
    )
    
    print(f"\nStage 3 Complete:")
    print(f"  Phase 3a (frozen) accuracy: {stage3a_acc:.2f}%")
    print(f"  Phase 3b (fine-tuned) accuracy: {stage3b_acc:.2f}%")
    
    # Save training history
    trainer.save_history(os.path.join(args.save_dir, 'stage3_history.json'))
    trainer.close()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print("MULTI-STAGE TRAINING COMPLETE")
    print(f"{'='*70}")
    if not args.skip_stage2:
        print(f"Stage 2 Final Accuracy: {stage2b_acc:.2f}%")
    print(f"Stage 3 Final Accuracy: {stage3b_acc:.2f}%")
    print(f"\nCheckpoints saved to: {args.save_dir}")
    print(f"TensorBoard logs saved to: {args.log_dir}")
    print(f"\nTo view TensorBoard:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-Stage Transfer Learning for Wildlife Classification'
    )
    
    # Data paths
    parser.add_argument('--stage2-image-dir', type=str, 
                       default='data/stage2/images',
                       help='Stage 2 (general wildlife) image directory')
    parser.add_argument('--stage2-metadata', type=str,
                       default='data/stage2/metadata.csv',
                       help='Stage 2 metadata CSV file')
    parser.add_argument('--stage3-image-dir', type=str,
                       default='data/stage3/images',
                       help='Stage 3 (target) image directory')
    parser.add_argument('--stage3-metadata', type=str,
                       default='data/stage3/metadata.csv',
                       help='Stage 3 metadata CSV file')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Stage 2 hyperparameters
    parser.add_argument('--stage2-epochs-frozen', type=int, default=15,
                       help='Stage 2 epochs with frozen backbone')
    parser.add_argument('--stage2-epochs-finetune', type=int, default=10,
                       help='Stage 2 epochs for fine-tuning')
    parser.add_argument('--stage2-lr-frozen', type=float, default=0.001,
                       help='Stage 2 learning rate (frozen)')
    parser.add_argument('--stage2-lr-finetune', type=float, default=0.0001,
                       help='Stage 2 learning rate (fine-tuning)')
    
    # Stage 3 hyperparameters
    parser.add_argument('--stage3-epochs-frozen', type=int, default=20,
                       help='Stage 3 epochs with frozen backbone')
    parser.add_argument('--stage3-epochs-finetune', type=int, default=15,
                       help='Stage 3 epochs for fine-tuning')
    parser.add_argument('--stage3-lr-frozen', type=float, default=0.001,
                       help='Stage 3 learning rate (frozen)')
    parser.add_argument('--stage3-lr-finetune', type=float, default=0.0001,
                       help='Stage 3 learning rate (fine-tuning)')
    
    # Other settings
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='runs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--skip-stage2', action='store_true',
                       help='Skip Stage 2 training')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Run training
    main(args)
```

---

## Evaluation and Analysis

### Step 10: Comprehensive Evaluation Script

Create `src/evaluation.py`:

```python
"""
Comprehensive evaluation and analysis for wildlife classification.

Provides:
- Detailed metrics (accuracy, precision, recall, F1)
- Confusion matrices
- Per-class analysis
- Comparison between model configurations
"""

import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix, 
                            precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from tqdm import tqdm

from model_architecture import create_model
from data_preprocessing import create_data_loaders


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_comprehensive(self, test_loader, class_names, save_dir='results'):
        """
        Complete evaluation with all metrics and visualizations.
        
        Args:
            test_loader: Test data loader
            class_names: List of class names
            save_dir: Directory to save results
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect predictions
        all_preds, all_labels, all_probs = self._collect_predictions(test_loader)
        
        # Calculate metrics
        results = {}
        
        # Overall accuracy
        accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
        results['accuracy'] = accuracy
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        
        # Top-5 accuracy
        top5_acc = self._calculate_top5_accuracy(all_probs, all_labels)
        results['top5_accuracy'] = top5_acc
        print(f"Top-5 Accuracy: {top5_acc:.2f}%")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Macro averages
        results['precision_macro'] = np.mean(precision)
        results['recall_macro'] = np.mean(recall)
        results['f1_macro'] = np.mean(f1)
        
        print(f"\nMacro-averaged metrics:")
        print(f"  Precision: {results['precision_macro']:.4f}")
        print(f"  Recall: {results['recall_macro']:.4f}")
        print(f"  F1-Score: {results['f1_macro']:.4f}")
        
        # Classification report
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        report = classification_report(
            all_labels, all_preds, target_names=class_names, digits=3
        )
        print(report)
        
        # Save report
        with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(
            cm, class_names, 
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # Per-class accuracy analysis
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        self.plot_per_class_accuracy(
            per_class_acc, class_names,
            save_path=os.path.join(save_dir, 'per_class_accuracy.png')
        )
        
        # Save detailed per-class results
        per_class_df = pd.DataFrame({
            'Species': class_names,
            'Accuracy': per_class_acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
        per_class_df = per_class_df.sort_values('Accuracy', ascending=False)
        per_class_df.to_csv(
            os.path.join(save_dir, 'per_class_metrics.csv'), index=False
        )
        
        print("\n" + "="*70)
        print("Top 5 Best Performing Species:")
        print(per_class_df.head(5).to_string(index=False))
        
        print("\nTop 5 Worst Performing Species:")
        print(per_class_df.tail(5).to_string(index=False))
        
        # Save results
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {save_dir}")
        
        return results
    
    def _collect_predictions(self, test_loader):
        """Collect all predictions and labels from test set."""
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\nCollecting predictions...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                if len(batch) == 3:
                    images, metadata, labels = batch
                    images = images.to(self.device)
                    metadata = metadata.to(self.device)
                    outputs = self.model(images, metadata)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    outputs = self.model(images, None)
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return all_preds, all_labels, all_probs
    
    def _calculate_top5_accuracy(self, probs, labels):
        """Calculate top-5 accuracy."""
        probs = np.array(probs)
        labels = np.array(labels)
        
        top5_preds = np.argsort(probs, axis=1)[:, -5:]
        correct = np.any(top5_preds == labels[:, np.newaxis], axis=1)
        
        return 100.0 * np.mean(correct)
    
    def plot_confusion_matrix(self, cm, class_names, save_path, normalize=False):
        """Plot confusion matrix."""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {save_path}")
    
    def plot_per_class_accuracy(self, accuracies, class_names, save_path):
        """Plot per-class accuracy bar chart."""
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)
        sorted_acc = accuracies[sorted_indices]
        sorted_names = [class_names[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if acc < 0.7 else 'orange' if acc < 0.85 else 'green' 
                 for acc in sorted_acc]
        plt.barh(range(len(sorted_names)), sorted_acc * 100, color=colors, alpha=0.7)
        plt.yticks(range(len(sorted_names)), sorted_names)
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.axvline(x=85, color='gray', linestyle='--', linewidth=1, 
                   label='85% threshold')
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Per-class accuracy plot saved to: {save_path}")


def compare_model_configurations(configs, test_loader, class_names, save_dir='results'):
    """
    Compare different model configurations for ablation study.
    
    Args:
        configs: Dict of {name: (model, include_metadata)} configurations
        test_loader: Test data loader
        class_names: List of class names
        save_dir: Directory to save comparison results
    """
    print("\n" + "="*70)
    print("MODEL CONFIGURATION COMPARISON")
    print("="*70)
    
    results = {}
    
    for name, (model, include_metadata) in configs.items():
        print(f"\nEvaluating: {name}")
        print("-" * 50)
        
        evaluator = ModelEvaluator(model)
        
        # Quick accuracy evaluation
        correct = 0
        total = 0
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=name):
                if len(batch) == 3 and include_metadata:
                    images, metadata, labels = batch
                    images = images.to(evaluator.device)
                    metadata = metadata.to(evaluator.device)
                    outputs = model(images, metadata)
                else:
                    images, labels = batch
                    images = images.to(evaluator.device)
                    outputs = model(images, None)
                
                labels = labels.to(evaluator.device)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        results[name] = accuracy
        print(f"Accuracy: {accuracy:.2f}%")
    
    # Plot comparison
    plot_model_comparison(results, save_path=os.path.join(save_dir, 'model_comparison.png'))
    
    # Save results
    with open(os.path.join(save_dir, 'ablation_study.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results


def plot_model_comparison(results, save_path):
    """Plot model comparison bar chart."""
    names = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(names)), accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.xticks(range(len(names)), names, rotation=15, ha='right')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Configuration Comparison', fontsize=14, fontweight='bold')
    plt.ylim([0, 100])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nModel comparison saved to: {save_path}")


if __name__ == '__main__':
    print("Evaluation module loaded successfully")
    print("Import this module to evaluate your trained models")
```

---

## Results Generation for Paper

### Step 11: Generate Results for Academic Paper

Create `src/generate_results.py`:

```python
"""
Generate results, tables, and figures for the academic paper.

Creates:
- LaTeX tables with performance metrics
- Training curve visualizations
- Ablation study results
- Metadata impact analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_training_curves(history_paths, labels, save_path='results/training_curves.png'):
    """
    Plot training and validation curves from multiple training runs.
    
    Args:
        history_paths: List of paths to training history JSON files
        labels: List of labels for each history
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (path, label) in enumerate(zip(history_paths, labels)):
        with open(path, 'r') as f:
            history = json.load(f)
        
        color = colors[i % len(colors)]
        
        # Loss curves
        axes[0].plot(history['train_loss'], label=f'{label} (Train)', 
                    color=color, linestyle='-', linewidth=2)
        axes[0].plot(history['val_loss'], label=f'{label} (Val)', 
                    color=color, linestyle='--', linewidth=2)
        
        # Accuracy curves
        axes[1].plot(history['train_acc'], label=f'{label} (Train)', 
                    color=color, linestyle='-', linewidth=2)
        axes[1].plot(history['val_acc'], label=f'{label} (Val)', 
                    color=color, linestyle='--', linewidth=2)
    
    # Loss subplot
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy subplot
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {save_path}")


def generate_results_table_latex(results_dict, save_path='results/results_table.tex'):
    """
    Generate LaTeX table for paper.
    
    Args:
        results_dict: Dictionary of {method_name: {metrics}} results
        save_path: Path to save LaTeX table
    """
    latex_table = r"""
\begin{table}[h]
\centering
\caption{Classification Performance Comparison}
\label{tab:results}
\begin{tabular}{lccc}
\hline
\textbf{Method} & \textbf{Accuracy (\%)} & \textbf{Top-5 Acc. (\%)} & \textbf{F1-Score} \\
\hline
"""
    
    for method, metrics in results_dict.items():
        acc = metrics.get('accuracy', 0)
        top5 = metrics.get('top5_accuracy', 0)
        f1 = metrics.get('f1_macro', 0)
        latex_table += f"{method} & {acc:.2f} & {top5:.2f} & {f1:.3f} \\\\\n"
    
    latex_table += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to: {save_path}")
    print("\nTable preview:")
    print(latex_table)
    
    return latex_table


def generate_ablation_table_latex(ablation_results, save_path='results/ablation_table.tex'):
    """
    Generate LaTeX table for ablation study.
    
    Args:
        ablation_results: Dictionary of {configuration: accuracy}
        save_path: Path to save LaTeX table
    """
    latex_table = r"""
\begin{table}[h]
\centering
\caption{Ablation Study Results}
\label{tab:ablation}
\begin{tabular}{lcccc}
\hline
\textbf{Method} & \textbf{Multi-Stage} & \textbf{Metadata} & \textbf{Accuracy (\%)} & \textbf{Improvement} \\
\hline
"""
    
    # Define configurations
    configs = [
        ("Single-stage, No Metadata", False, False),
        ("Single-stage, With Metadata", False, True),
        ("Multi-stage, No Metadata", True, False),
        ("Multi-stage, With Metadata (Ours)", True, True)
    ]
    
    baseline_acc = ablation_results.get("Single-stage, No Metadata", 0)
    
    for name, multi_stage, metadata in configs:
        acc = ablation_results.get(name, 0)
        improvement = acc - baseline_acc
        
        multi_stage_str = "\\checkmark" if multi_stage else "—"
        metadata_str = "\\checkmark" if metadata else "—"
        improvement_str = f"+{improvement:.2f}" if improvement > 0 else f"{improvement:.2f}"
        
        latex_table += f"{name} & {multi_stage_str} & {metadata_str} & {acc:.2f} & {improvement_str} \\\\\n"
    
    latex_table += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex_table)
    
    print(f"Ablation table saved to: {save_path}")
    print("\nTable preview:")
    print(latex_table)
    
    return latex_table


def analyze_metadata_impact(with_metadata_preds, without_metadata_preds, 
                           labels, class_names, save_dir='results'):
    """
    Analyze the specific contribution of metadata integration.
    
    Args:
        with_metadata_preds: Predictions with metadata
        without_metadata_preds: Predictions without metadata
        labels: True labels
        class_names: List of class names
        save_dir: Directory to save analysis
    """
    print("\n" + "="*70)
    print("METADATA IMPACT ANALYSIS")
    print("="*70)
    
    # Per-class accuracy with and without metadata
    per_class_with = []
    per_class_without = []
    
    for class_idx in range(len(class_names)):
        class_mask = np.array(labels) == class_idx
        
        if class_mask.sum() > 0:
            acc_with = np.mean(np.array(with_metadata_preds)[class_mask] == class_idx)
            acc_without = np.mean(np.array(without_metadata_preds)[class_mask] == class_idx)
        else:
            acc_with = 0
            acc_without = 0
        
        per_class_with.append(acc_with * 100)
        per_class_without.append(acc_without * 100)
    
    # Calculate improvement
    improvements = np.array(per_class_with) - np.array(per_class_without)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Species': class_names,
        'Accuracy (No Metadata)': per_class_without,
        'Accuracy (With Metadata)': per_class_with,
        'Improvement': improvements
    })
    df = df.sort_values('Improvement', ascending=False)
    
    # Save to CSV
    df.to_csv(f'{save_dir}/metadata_impact.csv', index=False)
    
    print("\nTop 10 species with highest improvement from metadata:")
    print(df.head(10).to_string(index=False))
    
    # Plot improvement
    plt.figure(figsize=(12, 8))
    colors = ['green' if x > 0 else 'red' for x in df['Improvement']]
    plt.barh(range(len(df)), df['Improvement'], color=colors, alpha=0.7)
    plt.yticks(range(len(df)), df['Species'])
    plt.xlabel('Accuracy Improvement (%)', fontsize=12)
    plt.title('Impact of Metadata Integration by Species', 
             fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metadata_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nMetadata impact analysis saved to: {save_dir}")
    
    # Overall statistics
    overall_improvement = np.mean(improvements)
    print(f"\nOverall average improvement: {overall_improvement:.2f}%")
    print(f"Species with improvement: {np.sum(improvements > 0)} / {len(class_names)}")
    print(f"Max improvement: {np.max(improvements):.2f}% ({class_names[np.argmax(improvements)]})")


if __name__ == '__main__':
    print("Results generation module loaded successfully")
    print("Use this module to create paper-ready figures and tables")
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors
**Problem:** GPU runs out of memory during training  
**Solutions:**
```python
# Reduce batch size
batch_size = 16  # or 8

# Enable gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

#### 2. Slow Training
**Problem:** Training takes too long  
**Solutions:**
```python
# Increase num_workers
num_workers = 8

# Use pin_memory
pin_memory = True

# Enable cudnn benchmarking
torch.backends.cudnn.benchmark = True
```

#### 3. Poor Convergence
**Problem:** Model doesn't learn or plateaus early  
**Solutions:**
- Check learning rate (try 1e-3, 1e-4, 1e-5)
- Verify data normalization matches ImageNet stats
- Ensure labels are correctly encoded
- Check for class imbalance (use weighted loss)
- Increase model capacity or training epochs

#### 4. Metadata Not Helping
**Problem:** Metadata integration doesn't improve performance  
**Solutions:**
- Verify cyclical encoding implementation
- Check metadata values are normalized correctly
- Ensure metadata is not corrupted or missing
- Try different fusion architectures
- Analyze which species benefit from metadata

---

## Quick Start Commands

### Train the Complete Pipeline
```bash
# Navigate to project directory
cd wildlife_classifier

# Activate environment
source wildlife_classifier_env/bin/activate

# Run full training pipeline
python src/main_training.py \
    --stage2-image-dir data/stage2/images \
    --stage2-metadata data/stage2/metadata.csv \
    --stage3-image-dir data/stage3/images \
    --stage3-metadata data/stage3/metadata.csv \
    --batch-size 32 \
    --stage2-epochs-frozen 15 \
    --stage2-epochs-finetune 10 \
    --stage3-epochs-frozen 20 \
    --stage3-epochs-finetune 15 \
    --save-dir models \
    --log-dir runs
```

### View Training Progress
```bash
# Launch TensorBoard
tensorboard --logdir runs

# Open browser to: http://localhost:6006
```

### Evaluate Trained Model
```python
from src.evaluation import ModelEvaluator
from src.model_architecture import create_model
from src.data_preprocessing import create_data_loaders

# Load model
model = create_model(num_classes=48)
checkpoint = torch.load('models/stage3b_finetune_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Load test data
_, test_loader, dataset = create_data_loaders(
    'data/stage3/images',
    'data/stage3/metadata.csv',
    include_metadata=True
)

# Evaluate
evaluator = ModelEvaluator(model)
results = evaluator.evaluate_comprehensive(
    test_loader,
    list(dataset.idx_to_species.values()),
    save_dir='results/final_evaluation'
)
```

---

## Expected Results

Based on the literature and your methodology:

### Baseline Comparisons
1. **Single-stage, No Metadata**: ~85-88% accuracy
2. **Single-stage, With Metadata**: ~86-89% accuracy (+0.5-1.0%)
3. **Multi-stage, No Metadata**: ~89-92% accuracy
4. **Multi-stage, With Metadata (Ours)**: ~91-94% accuracy (**target**)

### Key Findings to Highlight in Paper
1. Multi-stage transfer learning provides **2-4% improvement** over single-stage
2. Metadata integration adds **0.5-1.5% improvement** at each stage
3. Combined approach achieves **state-of-the-art** results
4. Certain species (nocturnal, seasonal) benefit more from metadata
5. Geographic metadata helps with habitat-specific species

---

## Timeline for Implementation

### Week 1: Setup and Data Preparation
- [ ] Environment setup
- [ ] Dataset acquisition and organization
- [ ] Metadata CSV creation
- [ ] Test data loading pipeline

### Week 2: Model Implementation
- [ ] Implement base architecture
- [ ] Add SE blocks
- [ ] Implement metadata encoder
- [ ] Test forward pass

### Week 3: Stage 2 Training
- [ ] Train general wildlife model
- [ ] Evaluate and checkpoint
- [ ] Analyze initial results

### Week 4: Stage 3 Training
- [ ] Train target-specific model
- [ ] Full fine-tuning
- [ ] Comprehensive evaluation

### Week 5: Analysis and Paper Writing
- [ ] Ablation studies
- [ ] Generate all figures and tables
- [ ] Write Methods section
- [ ] Write Results section

---

## Citation Information

When writing your paper, remember to cite:

1. **Norouzzadeh et al. (2018)** - Wildlife transfer learning baseline
2. **Liu et al. (2024)** - Temporal-SE-ResNet50 and metadata fusion
3. **Zhang et al. (2024)** - Multi-stage transfer learning framework
4. **He et al. (2016)** - ResNet architecture
5. **Hu et al. (2018)** - Squeeze-and-Excitation networks

---

## Conclusion

This guide provides a complete implementation of your multi-stage transfer learning with metadata integration methodology. Follow these steps systematically, starting with environment setup and progressing through each training stage.

**Key Success Factors:**
1. **Data quality**: Ensure metadata is accurate and complete
2. **Systematic training**: Follow the multi-stage pipeline exactly
3. **Proper evaluation**: Run comprehensive ablation studies
4. **Clear documentation**: Track all experiments and results

Good luck with your project! 🐾🦁🐘

---

**For questions or issues, refer to:**
- PyTorch documentation: https://pytorch.org/docs/
- TensorBoard guide: https://www.tensorflow.org/tensorboard
- Project proposal and literature review documents