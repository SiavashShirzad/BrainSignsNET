# BrainSignsNET — revised model (trained weights and inference)

This folder contains the model weights and the inference code used in the revised manuscript.

- **Input:** a raw 3D T1-weighted brain MRI scan in NIfTI format. Skull stripping, registration and cropping are not needed.
- **Output:** 23 anatomical landmarks in scanner (RAS) coordinates, in millimetres.

## Installation

```bash
pip install torch torchvision nibabel numpy pandas
```

## Usage

```bash
cd brainsignsnet_r2
python predict.py /path/to/scan.nii.gz -o landmarks.csv
python predict.py --list scans.csv -o landmarks.csv   # scans.csv: columns id,image_path
```

The output file has one row per landmark, with the columns `id, landmark, x, y, z`.

Options:

- `--device cpu` runs inference on the CPU.
- `--weights` uses another weight file.

On a GPU, inference takes about 0.16 s per scan, plus the time to load the scan.

## Landmarks

| | |
|---|---|
| Corpus callosum | posterior, anterior |
| Putamen, caudate, thalamus, pallidum, hippocampus, amygdala | left and right centres |
| Midline points | optic chiasm, intraventricular foramen, anterior commissure (AC), posterior commissure (PC) |
| Brainstem | midbrain, pons, medulla |
| Internal capsule | genu, left and right |

The order of the landmarks is given by `LANDMARKS` in `model.py`.

## Weights

`weights/brainsignsnet_r2_fp16.pth` holds the final model from the revised manuscript.

- **Training:** 2,391 scans from BLSA and BIOCARD. Labels for 21 landmarks come from FreeSurfer segmentations; AC and PC labels come from a network trained on manual AC/PC annotations.
- **Checkpoint:** the 21-landmark model extended to 23 landmarks (AC and PC added) and trained further; selected on a separate participant-level tuning set.
- **Format:** stored in half precision to keep the file under GitHub's size limit; they are converted back to float32 when loaded.
- **Accuracy:** on 15 test scans, predictions differed from those of the full-precision model by 0.002 mm on average (maximum 0.1 mm).

The weights can also serve as a starting point for fine-tuning on new landmarks. To fine-tune:

1. Load the state dictionary into `BrainSignsNET(num_keypoints=K)`.
2. Copy the output channels for the landmarks you keep.
3. Initialize the channels for new landmarks randomly.

## Preprocessing (done inside `predict.py`)

1. The scan is reoriented to RAS.
2. The largest physical dimension is resampled to 128 voxels, and the other dimensions keep the same voxel size.
3. The volume is centre-padded to 128³.
4. Intensities are z-score normalized.

Predicted grid coordinates are mapped back to the original scanner space through the image affine.

## Citation

If you use this model, please cite the BrainSignsNET article (Neuroinformatics, in revision).
