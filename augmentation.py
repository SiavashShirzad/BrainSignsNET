# augmentation.py
import numpy as np
import random
import torch
from scipy.ndimage import rotate, gaussian_filter, map_coordinates, zoom

class Augment3D:
    """
    A class for applying 3D augmentations to an image volume and keypoints.
    """
    def __init__(self,
                 flip_prob=0.5,
                 rotate_angle=20,       # degrees
                 intensity_shift=0.2,
                 crop_size=None,        
                 crop_prob=0.5,
                 shift_range=(30,30,30),
                 shift_prob=0.5,
                 noise_std=0.02,
                 noise_prob=0.5,
                 elastic_alpha=2.0,
                 elastic_sigma=8.0,
                 elastic_prob=0.5,
                 contrast_range=0.2,
                 contrast_prob=0.5,
                 brightness_range=0.2,
                 brightness_prob=0.5,
                 zoom_range=(0.9,1.1),
                 zoom_prob=0.5,
                 dropout_prob=0.5,
                 dropout_cube_fraction=0.1,
                 dropout_blocks=1,
                 salt_pepper_prob=0.5,
                 salt_pepper_amount=0.02,
                 speckle_prob=0.5,
                 speckle_std=0.05):
        self.flip_prob = flip_prob
        self.rotate_angle = rotate_angle
        self.intensity_shift = intensity_shift
        self.crop_size = crop_size
        self.crop_prob = crop_prob
        self.shift_range = shift_range
        self.shift_prob = shift_prob
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.zoom_range = zoom_range
        self.zoom_prob = zoom_prob
        self.dropout_prob = dropout_prob
        self.dropout_cube_fraction = dropout_cube_fraction
        self.dropout_blocks = dropout_blocks
        self.salt_pepper_prob = salt_pepper_prob
        self.salt_pepper_amount = salt_pepper_amount
        self.speckle_prob = speckle_prob
        self.speckle_std = speckle_std

    def __call__(self, volume, keypoints):
        """
        Apply augmentations to a volume and adjust keypoints.
        volume: torch.Tensor of shape [1, D, H, W]
        keypoints: torch.Tensor of shape (num_keypoints*3,)
        """
        vol_np = volume.squeeze(0).numpy()
        kp = keypoints.cpu().numpy().reshape(-1, 3)
        D, H, W = vol_np.shape
        
        # Example: random flip along the z-axis.
        if random.random() < self.flip_prob:
            vol_np = np.flip(vol_np, axis=0).copy()
            kp[:, 0] = (D - 1) - kp[:, 0]
        # (Additional augmentations such as rotation, translation, etc. would be added here.)

        volume_aug = torch.from_numpy(vol_np).unsqueeze(0).float()
        keypoints_aug = torch.from_numpy(kp.reshape(-1)).float()
        return volume_aug, keypoints_aug
