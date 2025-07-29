# Medical Image Segmentation with Rollas U‑Net

A PyTorch-based pipeline for segmenting medical images (256×256) using a U‑Net variant (“RollasUnet”).  
Designed to run in a Kaggle or Colab environment with the `a0-2025-medical-image-segmentation` dataset.


## 📁 Dataset Structure

Assumes the following folder layout under `DATASET_PATH` (defaults to `/kaggle/input/a0-2025-medical-image-segmentation/Dataset`):

```

Dataset/
├── Train/
│   ├── Image/        # .jpg files
│   └── Mask/         # .png files (binary masks)
└── Test/
└── Image/        # .jpg files for inference

````

Each mask file name matches its image name (e.g. `0001.png` ↔ `0001.jpg`).


## ⚙️ Requirements

```bash
pip install \
  numpy pandas opencv-python pillow matplotlib tqdm \
  torch torchvision albumentations
````


## 🔧 Configuration

Edit at top of script or notebook:

```python
DATASET_PATH   = "/kaggle/input/a0-2025-medical-image-segmentation/Dataset"
IMAGE_HEIGHT   = 256
IMAGE_WIDTH    = 256
BATCH_SIZE     = 8
NUM_WORKERS    = 2
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 25
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
```


## 🚀 Usage

1. **Data loading & visualization**
   The script will:

   * Enumerate `Train/Mask` ↔ `Train/Image` pairs
   * Shuffle & split 80/20 into train/validation
   * Display a sample image+mask pair
   * Check for GPU availability

2. **Dataset & DataLoader**

   * `SegmentationDataset` applies Albumentations transforms
   * `train_loader` / `val_loader` yield `(image, mask)` batches

3. **Model definition & sanity check**

   * `RollasUnet` implements a standard U‑Net with `DoubleConv` blocks
   * Sanity‑check via a dummy forward pass

4. **Training loop**

   * Uses `BCEWithLogitsLoss` (optionally combined with Dice loss)
   * Tracks train/val loss and validation Dice coefficient
   * Saves best model to `best_model.pth`

5. **Visualization**

   * `visualize_predictions(model, val_loader, device)` shows a few (input, ground truth, predicted) triplets

---

   * `TestDataset` loads and resizes test images
   * `generate_submission_and_visualize(...)` runs model on test set, resizes masks back to original dimensions, and displays a handful of results
