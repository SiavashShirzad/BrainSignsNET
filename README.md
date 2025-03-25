# 3D Medical Image Keypoint Detection

## 📌 Overview
This repository provides a comprehensive pipeline for detecting anatomical keypoints in 3D medical images using a deep learning model based on 3D ResNet and UNet architectures.

## 🚀 Features
- **3D Data Augmentation:** Robust set of augmentations for medical volumes.
- **Flexible Dataset:** Custom PyTorch dataset for handling NIfTI (.nii.gz) images and keypoints.
- **Advanced Model:** 3D ResNet backbone combined with a UNet decoder.
- **Training and Validation:** Detailed scripts for training, validation, metrics tracking, and checkpointing.
- **Visualization:** Clear visualizations of predictions versus ground truth.
- **Inference Pipeline:** Easy-to-use inference script to predict keypoints and save results to CSV.

## 📂 Project Structure
```
project/
├── augmentation.py      # 3D data augmentation methods
├── dataset.py           # Data loading and handling
├── model.py             # Model architecture definition
├── training.py          # Training script
├── validation.py        # Validation script
├── utils.py             # Utility functions for data processing
├── visualization.py     # Visualization scripts
├── predict.py           # Inference and prediction script
├── main.py              # Main training and validation entry point
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🔧 Installation
**Clone the repository:**
```bash
git clone https://github.com/SiavashShirzad/BrainSignsNET.git
cd BrainSignsNET
```

**Set up a Python virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
.\venv\Scripts\activate  # On Windows
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🗃️ Data Preparation
Prepare a CSV file with columns:
- `image_path`: Path to each NIfTI file (.nii.gz).
- Keypoint columns (ending in `_center`), containing coordinates normalized to [0, 1].

**Example CSV structure:**
```csv
id,image_path,CorpusCallosum_center,LeftHippocampus_center,...
001,data/patient001.nii.gz,"[0.5, 0.3, 0.6]","[0.4, 0.5, 0.2]",...
002,data/patient002.nii.gz,"[0.6, 0.2, 0.5]","[0.5, 0.4, 0.3]",...
```

## 🏃 Training the Model
Use the main training script:
```bash
python main.py --csv path/to/data.csv --pid_column id --epochs 50
```
- **`--csv`**: Path to CSV data file.
- **`--pid_column`**: Column to group samples (e.g., patient id).
- **`--epochs`**: Number of epochs to train.

## 🎯 Inference & Prediction
Predict keypoints on a new image:
```bash
python predict.py --image path/to/new_image.nii.gz --checkpoint best_model.pth --output_csv predicted_coords.csv
```
- **`--image`**: Path to input image.
- **`--checkpoint`**: Path to trained model weights.
- **`--output_csv`**: CSV file for predicted coordinates.

## 📈 Visualization
Use `visualization.py` functions in notebooks or scripts to visualize model predictions clearly.

## 📚 Requirements
Dependencies (auto-install with `requirements.txt`):
```bash
torch
torchvision
numpy
pandas
matplotlib
scipy
nibabel
scikit-learn
opencv-python
tqdm
```

## 🙏 Acknowledgements
- [PyTorch](https://pytorch.org/)
- [Nibabel](https://nipy.org/nibabel/)
- [Scikit-learn](https://scikit-learn.org/)
- [OpenCV](https://opencv.org/)
- [TQDM](https://github.com/tqdm/tqdm)
