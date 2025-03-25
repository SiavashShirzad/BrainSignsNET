# visualization.py
import matplotlib.pyplot as plt
import cv2
import numpy as np

def visualize_sample(sample, keyPoints):
    """
    Visualize a single sample showing the image slices with overlaid heatmaps and keypoint markers.
    """
    image = sample["image"][0].cpu().numpy()       # (128, 128, 128)
    heatmaps = sample["heatmaps"][0].cpu().numpy()   # (num_keypoints, 128, 128, 128)
    coords = sample["coords"][0].cpu().numpy().reshape(-1, 3)
    num_keypoints = coords.shape[0]
    
    for i in range(num_keypoints):
        x, y, z = coords[i]
        x_slice = int(round(x))
        y_slice = int(round(y))
        z_slice = int(round(z))
        hm = heatmaps[i]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Sagittal view (fix x).
        axes[0].imshow(image[x_slice, :, :], cmap='gray', origin='lower')
        axes[0].imshow(hm[x_slice, :, :], cmap='hot', alpha=0.5, origin='lower')
        axes[0].scatter(z_slice, y_slice, color='cyan', marker='x')
        axes[0].set_title(f"{keyPoints[i]} Sagittal (x = {x_slice})")
        
        # Coronal view (fix y).
        axes[1].imshow(image[:, y_slice, :].T, cmap='gray', origin='lower')
        axes[1].imshow(hm[:, y_slice, :].T, cmap='hot', alpha=0.5, origin='lower')
        axes[1].scatter(x_slice, z_slice, color='cyan', marker='x')
        axes[1].set_title(f"{keyPoints[i]} Coronal (y = {y_slice})")
        
        # Axial view (fix z).
        axes[2].imshow(image[:, :, z_slice].T, cmap='gray', origin='lower')
        axes[2].imshow(hm[:, :, z_slice].T, cmap='hot', alpha=0.5, origin='lower')
        axes[2].scatter(x_slice, y_slice, color='cyan', marker='x')
        axes[2].set_title(f"{keyPoints[i]} Axial (z = {z_slice})")
        
        plt.tight_layout()
        plt.show()
