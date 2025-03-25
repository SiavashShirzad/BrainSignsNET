# validation.py
import torch
import torch.nn as nn
from tqdm import tqdm

def validate_model(model, val_loader, device):
    """
    Evaluate the model on a validation set and print the average loss.
    """
    mse_loss = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['image'].to(device)
            gt_heatmaps = batch['heatmaps'].to(device)
            gt_attn_heatmap = batch['attn_heatmap'].to(device)
            gt_coords = batch['coords'].to(device)
            
            pred_attn, pred_heatmaps, pred_coords = model(images)
            coord_loss = mse_loss(pred_coords, gt_coords)
            heatmap_loss = mse_loss(pred_heatmaps, gt_heatmaps)
            attn_loss = mse_loss(pred_attn, gt_attn_heatmap)
            loss = coord_loss + heatmap_loss + attn_loss
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    avg_loss = total_loss / total_samples
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss
