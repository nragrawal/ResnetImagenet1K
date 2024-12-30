# ResNet-50 ImageNet Classifier

This model is a ResNet-50 trained on ImageNet dataset.

## Setup Instructions

### 1. Download ImageNet from Kaggle

```bash
Install kaggle CLI
pip install kaggle
Place kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
Download ImageNet
kaggle competitions download -c imagenet-object-localization-challenge
unzip imagenet-object-localization-challenge.zip -d ILSVRC
```

### 2. Prepare 10% Dataset

Create subset with 10% of classes

```bash
python prepare_imagenet_subset.py \
--source_dir ILSVRC \
--target_dir ILSVRC_subset_10 \
--subset_percentage 10
```

Organize validation set

```bash
python organize_validation.py \
--val_dir ILSVRC_subset_10/Data/CLS-LOC/val \
--devkit_dir ILSVRC_subset_10/Data/devkit
```

Verify dataset structure

```bash
python verify_dataset.py --data_dir ILSVRC_subset_10
```

### 3. Train and Verify on 10% Dataset

```bash
# Train on 10% dataset
python train_resnet.py \
    --data_dir ILSVRC_subset_10 \
    --batch_size 256 \
    --epochs 90 \
    --lr 0.1

# Monitor training
tail -f outputs/training_log_*.txt
```

### 4. Prepare 100% Dataset

```bash
# Create full dataset
python prepare_imagenet_subset.py \
    --source_dir ILSVRC \
    --target_dir ILSVRC_full \
    --subset_percentage 100

# Organize validation set
python organize_validation.py \
    --val_dir ILSVRC_full/Data/CLS-LOC/val \
    --devkit_dir ILSVRC_full/Data/devkit
```

### 5. Train and Verify on Full Dataset

```bash
# Verify full dataset structure
python verify_dataset.py --data_dir ILSVRC_full
python verify_validation.py --val_dir ILSVRC_full/Data/CLS-LOC/val

# Train on full dataset
python train_resnet.py \
    --data_dir ILSVRC_full \
    --batch_size 256 \
    --epochs 90 \
    --lr 0.1
```

## Model Description

### Architecture
```
Input Image (224x224x3)
      ↓
Conv1 (7x7, 64, /2)
      ↓
MaxPool (3x3, /2)
      ↓
Stage1 (64-256) × 3  ←←← Skip Connection
[
  1×1, 64
  3×3, 64
  1×1, 256
]
      ↓
Stage2 (128-512) × 4  ←←← Skip Connection
[
  1×1, 128
  3×3, 128
  1×1, 512
]
      ↓
Stage3 (256-1024) × 6  ←←← Skip Connection
[
  1×1, 256
  3×3, 256
  1×1, 1024
]
      ↓
Stage4 (512-2048) × 3  ←←← Skip Connection
[
  1×1, 512
  3×3, 512
  1×1, 2048
]
      ↓
AvgPool
      ↓
FC (1000)
      ↓
Output (1000 classes)
```

### Skip Connection Detail
```
Input
  ↓
F(x) [Conv → BN → ReLU → Conv → BN]
  ↓         ↘
  ↓           Identity (x)
  ↓         ↙
Output = F(x) + x
```

- **Model type:** ResNet-50
- **Training Data:** ImageNet
- **Framework:** PyTorch
- **License:** MIT

## Training procedure

- **Training type:** Image Classification
- **Optimizer:** SGD
- **Learning rate:** 0.1
- **Batch size:** 256
- **Number of epochs:** 90

## Metrics

- Top-1 Accuracy: XX%
- Top-5 Accuracy: XX% 

## Scripts Documentation

### prepare_imagenet_subset.py

This script creates a subset of ImageNet for training:

```bash
python prepare_imagenet_subset.py \
    --source_dir ILSVRC \
    --target_dir ILSVRC_subset_10 \
    --subset_percentage 10
```

**Parameters:**
- `source_dir`: Path to original ImageNet dataset
- `target_dir`: Where to save the subset
- `subset_percentage`: Percentage of classes to include (10 or 100)

**What it does:**
1. Randomly selects N% of ImageNet classes
2. Creates directory structure for subset
3. Copies selected class images from train and val sets
4. Preserves ImageNet folder structure
5. Creates dataset statistics file

### train_resnet.py

Main training script with mixed precision and multi-GPU support:

```bash
python train_resnet.py \
    --data_dir ILSVRC_subset_10 \
    --batch_size 256 \
    --epochs 90 \
    --lr 0.1
```

**Parameters:**
- `data_dir`: Path to dataset
- `batch_size`: Batch size per GPU
- `epochs`: Number of training epochs
- `lr`: Initial learning rate
- `pretrained`: Use pretrained weights (optional)
- `resume`: Path to checkpoint for resuming training

**Features:**
1. **Mixed Precision Training**
   - Uses torch.cuda.amp for faster training
   - Automatically handles FP16/FP32 conversion

2. **Multi-GPU Support**
   - Automatic DataParallel for multiple GPUs
   - Scales batch size and learning rate
   - Adjusts number of workers

3. **Training Features**
   - Cosine learning rate scheduling
   - Top-1 and Top-5 accuracy metrics
   - Checkpoint saving and resuming
   - Progress bars with live metrics
   - Detailed logging to file

4. **Data Augmentation**
   - Random resized crops
   - Horizontal flips
   - Color jittering
   - Normalization using dataset statistics

**Output Structure:**
```
outputs/
├── training_log_YYYYMMDD_HHMMSS.txt  # Training logs
└── checkpoints/
    ├── checkpoint_epoch_001.pth       # Regular checkpoints
    ├── checkpoint_epoch_002.pth
    └── model_best.pth                 # Best model weights
```

**Monitoring Training:**
```bash
# View live training progress
tail -f outputs/training_log_*.txt

# Check GPU usage
nvidia-smi -l 1
``` 