# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs, device, checkpoint_path='best_model.pth'):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    mse_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        num_samples = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = batch['image'].to(device)
            gt_heatmaps = batch['heatmaps'].to(device)
            gt_attn_heatmap = batch['attn_heatmap'].to(device)
            gt_coords = batch['coords'].to(device)
            
            optimizer.zero_grad()
            pred_attn, pred_heatmaps, pred_coords = model(images)
            
            coord_loss = mse_loss(pred_coords, gt_coords)
            heatmap_loss = mse_loss(pred_heatmaps, gt_heatmaps)
            attn_loss = mse_loss(pred_attn, gt_attn_heatmap)
            total_loss = coord_loss + heatmap_loss + attn_loss
            total_loss.backward()
            optimizer.step()
            
            batch_size = images.size(0)
            running_loss += total_loss.item() * batch_size
            num_samples += batch_size
        
        epoch_loss = running_loss / num_samples
        print(f"Training Loss: {epoch_loss:.4f}")
        
        # Validation loop.
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images = batch['image'].to(device)
                gt_heatmaps = batch['heatmaps'].to(device)
                gt_attn_heatmap = batch['attn_heatmap'].to(device)
                gt_coords = batch['coords'].to(device)
                
                pred_attn, pred_heatmaps, pred_coords = model(images)
                coord_loss = mse_loss(pred_coords, gt_coords)
                heatmap_loss = mse_loss(pred_heatmaps, gt_heatmaps)
                attn_loss = mse_loss(pred_attn, gt_attn_heatmap)
                total_loss = coord_loss + heatmap_loss + attn_loss
                
                batch_size = images.size(0)
                val_loss += total_loss.item() * batch_size
                val_samples += batch_size
        epoch_val_loss = val_loss / val_samples
        print(f"Validation Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved best model.")
            
        scheduler.step()
    
    return model
