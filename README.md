# BrainSignsNET

BrainSignsNET localizes 23 anatomical landmarks on raw 3D T1-weighted brain MRI scans.

- It works in a single pass over the whole scan.
- Skull stripping, registration and cropping are not needed.
- Coordinates are returned in scanner (RAS) space, in millimetres.

This repository contains the model, the released trained weights, the inference script, and the code used for training and for extracting landmarks from FreeSurfer segmentations.

## Installation

```bash
git clone https://github.com/SiavashShirzad/BrainSignsNET.git
cd BrainSignsNET
pip install -r requirments.txt
```

## Landmark prediction

```bash
python predict.py /path/to/scan.nii.gz -o landmarks.csv
python predict.py --list scans.csv -o landmarks.csv      # scans.csv: columns id,image_path
```

The output has one row per landmark, with the columns `id, landmark, x, y, z`.

| Option | Use |
|---|---|
| `--device cpu` | run on the CPU |
| `--weights` | use another weight file |

On a GPU, inference takes about 0.16 s per scan.

## Landmarks

| Group | Landmarks |
|---|---|
| Corpus callosum | anterior, posterior |
| Commissures and ventricles | anterior commissure (AC), posterior commissure (PC), intraventricular foramen |
| Optic chiasm | optic chiasm |
| Brainstem | midbrain, pons, medulla |
| Basal ganglia | left and right putamen, caudate and pallidum |
| Thalamus | left and right thalamus |
| Medial temporal lobe | left and right hippocampus and amygdala |
| Internal capsule | left and right genu |

The output order is given by `LANDMARKS` in `model.py`.

## Trained weights

`weights/brainsignsnet_fp16.pth` holds the model described in the article.

- **Training data:** 2,391 BLSA and BIOCARD scans.
- **Checkpoint:** selected on a separate participant-level tuning set.
- **Labels:**
  - 21 landmarks come from FreeSurfer segmentations (`FSKeypointExtraction/`).
  - AC and PC come from a network trained on manual annotations.
- **Format:** stored in half precision to keep the file small; `model.load()` converts them back to float32.

The weights can also be used as a starting point for fine-tuning on other landmarks:

1. Load the state dictionary into `Keypoint3DResNetUNet(num_keypoints=K)`.
2. Keep the output channels of the landmarks you need.
3. Initialize the channels of new landmarks randomly.

### Preprocessing (inside `predict.py`)

1. Reorient the scan to RAS.
2. Resample the largest physical dimension to 128 voxels, keeping the same voxel size in every direction.
3. Centre-pad to 128³.
4. Apply z-score intensity normalization.

## Training

`main.py` trains the model from a CSV file that has an `image_path` column and one `<Landmark>_center` column per landmark, holding coordinates normalized to [0, 1]:

```bash
python main.py --csv path/to/data.csv --pid_column id --epochs 50
```

Labels can be generated from FreeSurfer `aparc+aseg` segmentations with `FSKeypointExtraction/extraction.py`.

## Repository structure

```
├── model.py                  # Network (Keypoint3DResNetUNet), landmark list and weight loader
├── predict.py                # Landmark prediction on raw NIfTI scans
├── weights/                  # Released trained weights
├── main.py, training.py, validation.py, dataset.py, augmentation.py, utils.py, visualization.py
└── FSKeypointExtraction/     # Landmark extraction from FreeSurfer segmentations
```

## Citation

If you use BrainSignsNET, please cite the BrainSignsNET article (Neuroinformatics, under revision).
