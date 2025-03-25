# utils.py
import numpy as np
from scipy.ndimage import gaussian_filter

def add_gaussian_to_volume(volume_shape, center, sigma):
    """
    Create a 3D heatmap (Gaussian blob) with a blob centered at 'center'.
    """
    heatmap = np.zeros(volume_shape, dtype=np.float32)
    z, y, x = center
    z_idx, y_idx, x_idx = int(round(z)), int(round(y)), int(round(x))
    if (0 <= z_idx < volume_shape[0] and 0 <= y_idx < volume_shape[1] and 0 <= x_idx < volume_shape[2]):
        heatmap[z_idx, y_idx, x_idx] = 1.0

    heatmap = gaussian_filter(heatmap, sigma=sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def load_nifti_image(image_path):
    """
    Load a NIfTI image and return the image data and voxel spacing.
    """
    import nibabel as nib
    img = nib.load(image_path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata().astype(np.float32)
    spacing = img.header.get_zooms()[:3]
    return data, spacing

def center_pad(volume, target_shape):
    """
    Center-pad a volume (assumed to be a NumPy array) to a target shape.
    """
    import torch
    import torch.nn.functional as F
    vol = torch.tensor(volume).unsqueeze(0).unsqueeze(0)
    current_shape = vol.shape[2:]
    pad_vals = [target_shape[i] - current_shape[i] for i in range(len(target_shape))]
    # Calculate symmetric padding per axis (in reverse order for F.pad)
    padding = []
    for p in reversed(pad_vals):
        pad_before = p // 2
        pad_after = p - pad_before
        padding.extend([pad_before, pad_after])
    padded = F.pad(vol, padding)
    return padded.squeeze(0)
