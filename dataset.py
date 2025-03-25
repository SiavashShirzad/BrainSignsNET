# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import nibabel as nib
import ast
import torch.nn.functional as F
from utils import add_gaussian_to_volume

class KeypointDFDataset(Dataset):
    """
    Dataset that loads 3D NIfTI images and keypoint data from a DataFrame.
    """
    def __init__(self, df, volume_size=(128,128,128), heatmap_size=(64,64,64),
                 augmentation=None, heatmap_sigma=2, keypoint_cols=None):
        self.df = df.reset_index(drop=True)
        self.volume_size = volume_size
        self.heatmap_size = heatmap_size
        self.augmentation = augmentation
        self.heatmap_sigma = heatmap_sigma
        if keypoint_cols is None:
            self.keypoint_cols = [col for col in df.columns if col.endswith('_center')]
        else:
            self.keypoint_cols = keypoint_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        # Load image
        img = nib.load(image_path)
        img = nib.as_closest_canonical(img)
        data = img.get_fdata().astype(np.float32)
        spacing = np.array(img.header.get_zooms()[:3])
        
        # For simplicity, we use F.interpolate to resize to the target volume size.
        volume = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        volume_resized = F.interpolate(volume, size=self.volume_size, mode='trilinear', align_corners=False)
        volume_resized = volume_resized.squeeze(0)
        
        # Process keypoints.
        keypoints = []
        for kp_col in self.keypoint_cols:
            try:
                kp_norm = np.array(ast.literal_eval(row[kp_col]), dtype=float)
            except Exception:
                kp_norm = np.array([0, 0, 0])
            # Convert normalized coordinates to voxel coordinates.
            kp_vox = kp_norm * np.array(self.volume_size)
            kp_vox = np.clip(kp_vox, 0, np.array(self.volume_size)-1)
            keypoints.extend(kp_vox.tolist())
        coords = torch.tensor(keypoints, dtype=torch.float)
        
        # Apply augmentation if provided.
        if self.augmentation is not None:
            volume_resized, coords = self.augmentation(volume_resized, coords)
        
        # Generate heatmaps.
        heatmaps_list = []
        for i in range(0, len(coords), 3):
            kp = coords[i:i+3].cpu().numpy()
            kp_scaled = (kp / np.array(self.volume_size)) * np.array(self.heatmap_size)
            kp_scaled = np.round(kp_scaled).astype(int)
            kp_scaled = np.clip(kp_scaled, 0, np.array(self.heatmap_size)-1)
            hm = add_gaussian_to_volume(self.heatmap_size, tuple(kp_scaled), self.heatmap_sigma)
            heatmaps_list.append(hm)
        heatmaps = torch.tensor(np.stack(heatmaps_list, axis=0), dtype=torch.float)
        attn_heatmap = torch.max(heatmaps, dim=0, keepdim=True)[0]
        
        sample = {
            "image": volume_resized,
            "heatmaps": heatmaps,
            "attn_heatmap": attn_heatmap,
            "coords": coords,
            "target_spacing": torch.tensor(spacing, dtype=torch.float32)
        }
        return sample

def get_dataloader_from_df(df, augmentation=None, batch_size=4, shuffle=True, num_workers=4):
    dataset = KeypointDFDataset(df, augmentation=augmentation)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def split_train_test(df, pid_column, test_size=0.2, random_state=42):
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[pid_column]))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    return train_df, test_df
