# predict.py
import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
from model import Keypoint3DResNetUNet
from utils import load_nifti_image
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_shape=(128,128,128)):
    """
    Load and resize a NIfTI image to a target shape.
    """
    data, spacing = load_nifti_image(image_path)
    import torch
    vol = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    vol_resized = F.interpolate(vol, size=target_shape, mode='trilinear', align_corners=False)
    vol_resized = vol_resized.squeeze(0)
    return vol_resized, spacing

def visualize_prediction(image, heatmaps, coords, keyPoints):
    """
    Visualize predicted keypoint heatmaps and overlay keypoint coordinates.
    """
    img = image[0].cpu().numpy()  # (128, 128, 128)
    heatmaps_np = heatmaps.cpu().numpy()
    coords_np = coords.cpu().numpy().reshape(-1, 3)
    num_keypoints = coords_np.shape[0]
    for i in range(num_keypoints):
        x, y, z = coords_np[i]
        x_slice = int(round(x))
        y_slice = int(round(y))
        z_slice = int(round(z))
        hm = heatmaps_np[i]
        fig, axes = plt.subplots(1, 3, figsize=(18,6))
        axes[0].imshow(img[x_slice,:,:], cmap='gray', origin='lower')
        axes[0].imshow(hm[x_slice,:,:], cmap='hot', alpha=0.5, origin='lower')
        axes[0].scatter(z_slice, y_slice, color='cyan', marker='x')
        axes[0].set_title(f"{keyPoints[i]} Sagittal")
        
        axes[1].imshow(img[:,y_slice,:].T, cmap='gray', origin='lower')
        axes[1].imshow(hm[:,y_slice,:].T, cmap='hot', alpha=0.5, origin='lower')
        axes[1].scatter(x_slice, z_slice, color='cyan', marker='x')
        axes[1].set_title(f"{keyPoints[i]} Coronal")
        
        axes[2].imshow(img[:,:,z_slice].T, cmap='gray', origin='lower')
        axes[2].imshow(hm[:,:,z_slice].T, cmap='hot', alpha=0.5, origin='lower')
        axes[2].scatter(x_slice, y_slice, color='cyan', marker='x')
        axes[2].set_title(f"{keyPoints[i]} Axial")
        plt.tight_layout()
        plt.show()

def save_coords_to_csv(coords, output_csv, keyPoints):
    """
    Save predicted keypoint coordinates to a CSV file.
    """
    coords_np = coords.cpu().numpy().reshape(-1, 3)
    df = pd.DataFrame(coords_np, columns=['z', 'y', 'x'])
    df['keypoint'] = keyPoints
    df.to_csv(output_csv, index=False)
    print(f"Coordinates saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Predict keypoints from an input image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input NIfTI image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--output_csv", type=str, default="predicted_coords.csv", help="CSV file for saving coordinates.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define keypoints (update the list as needed)
    keyPoints = [
        "Corpus Callosum Posterior",
        "Corpus Callosum Anterior",
        "Right Putamen",
        "Left Putamen",
        "Left Caudate",
        "Left Thalamus",
        "Left Pallidum",
        "Left Hippocampus",
        "Left Amygdala",
        "Right Caudate",
        "Right Thalamus",
        "Right Pallidum",
        "Right Hippocampus",
        "Right Amygdala",
        "Optic Chiasm",
        "Intraventricular Foramen",
        "Midbrain",
        "Pons",
        "Medulla",
        "Posterior Commissure",
        "Anterior Commissure",
        "Left Internal Capsule Genu",
        "Right Internal Capsule Genu"
    ]
    
    # Preprocess image.
    image_tensor, spacing = preprocess_image(args.image)
    image_tensor = image_tensor.to(device)
    
    # Load model.
    model = Keypoint3DResNetUNet(num_keypoints=len(keyPoints))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        pred_attn, pred_heatmaps, pred_coords = model(image_tensor.unsqueeze(0))
    
    # Visualize predictions.
    visualize_prediction(image_tensor, pred_heatmaps[0], pred_coords[0], keyPoints)
    
    # Save predicted coordinates.
    save_coords_to_csv(pred_coords[0], args.output_csv, keyPoints)

if __name__ == "__main__":
    main()
