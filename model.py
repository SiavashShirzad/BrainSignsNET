# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

def soft_argmax_3d(heatmaps):
    """
    Compute the soft-argmax over 3D heatmaps.
    """
    N, K, D, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(N, K, -1)
    softmax = F.softmax(heatmaps_flat, dim=2)
    device = heatmaps.device
    z_range = torch.linspace(0, D - 1, D, device=device)
    y_range = torch.linspace(0, H - 1, H, device=device)
    x_range = torch.linspace(0, W - 1, W, device=device)
    zz, yy, xx = torch.meshgrid(z_range, y_range, x_range, indexing='ij')
    zz = zz.contiguous().view(-1)
    yy = yy.contiguous().view(-1)
    xx = xx.contiguous().view(-1)
    exp_z = torch.sum(softmax * zz, dim=2)
    exp_y = torch.sum(softmax * yy, dim=2)
    exp_x = torch.sum(softmax * xx, dim=2)
    coords = torch.stack([exp_z, exp_y, exp_x], dim=2)
    coords = coords.view(N, -1)
    return coords

class Keypoint3DResNetUNet(nn.Module):
    """
    A simplified 3D keypoint detection model based on a ResNet backbone with upsampling.
    """
    def __init__(self, num_keypoints=2, pretrained=False, freeze_backbone=False):
        super(Keypoint3DResNetUNet, self).__init__()
        self.num_keypoints = num_keypoints
        backbone = r3d_18(pretrained=pretrained)
        backbone.fc = nn.Identity()
        old_conv = backbone.stem[0]
        new_conv = nn.Conv3d(
            in_channels=old_conv.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=(2, 2, 2),
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight
        backbone.stem[0] = new_conv
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.backbone = backbone

        # For simplicity, we use a single upsampling branch.
        self.up = nn.ConvTranspose3d(512, num_keypoints, kernel_size=2, stride=2)
        self.heatmap_head = nn.Conv3d(num_keypoints, num_keypoints, kernel_size=1)
    
    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1, 1)
        stem = self.backbone.stem(x)
        out1 = self.backbone.layer1(stem)
        out2 = self.backbone.layer2(out1)
        out3 = self.backbone.layer3(out2)
        out4 = self.backbone.layer4(out3)
        up = self.up(out4)
        heatmaps = self.heatmap_head(up)
        coords = soft_argmax_3d(heatmaps)
        # Create a dummy attention heatmap for compatibility.
        attn_heatmap = heatmaps.mean(dim=1, keepdim=True)
        return attn_heatmap, heatmaps, coords
