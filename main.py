# main.py
import argparse
import pandas as pd
from dataset import get_dataloader_from_df, split_train_test
from augmentation import Augment3D
from model import Keypoint3DResNetUNet
from training import train_model
from validation import validate_model
import torch

def main():
    parser = argparse.ArgumentParser(description="Train and validate a 3D keypoint detection model.")
    parser.add_argument("--csv", type=str, required=True, help="CSV file with image paths and keypoint data.")
    parser.add_argument("--pid_column", type=str, default="id", help="Column name for grouping (e.g., patient id).")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    args = parser.parse_args()
    
    # Load CSV.
    df = pd.read_csv(args.csv)
    train_df, val_df = split_train_test(df, args.pid_column)
    
    # Set up augmentation.
    augmentation = Augment3D(flip_prob=0.5, rotate_angle=20)
    
    train_loader = get_dataloader_from_df(train_df, augmentation=augmentation, batch_size=4)
    val_loader = get_dataloader_from_df(val_df, augmentation=None, batch_size=4)
    
    # Determine number of keypoints from the CSV.
    keypoint_cols = [col for col in df.columns if col.endswith('_center')]
    num_keypoints = len(keypoint_cols)
    
    # Create model.
    model = Keypoint3DResNetUNet(num_keypoints=num_keypoints)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model.
    model = train_model(model, train_loader, val_loader, num_epochs=args.epochs, device=device, checkpoint_path="best_model.pth")
    
    # Validate the model.
    validate_model(model, val_loader, device)

if __name__ == "__main__":
    main()
