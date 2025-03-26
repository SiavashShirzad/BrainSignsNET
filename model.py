import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

def soft_argmax_3d(heatmaps):
    """
    Compute the soft argmax for 3D heatmaps.

    :param heatmaps: Tensor of shape (N, K, D, H, W)
    :return: Tensor of shape (N, K*3) with the (z, y, x) coordinates arranged as
             [z1, y1, x1, z2, y2, x2, ...]
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
    A multi-task 3D keypoint model with UNet-style skip connections.

    Encoder: 3D ResNet-18 (with modified stem to downsample depth)
       - Produces intermediate features: stem, layer1, layer2, layer3, layer4.

    Two Decoders:
      1. Multi-class decoder for keypoint heatmaps.
      2. Attention decoder for a single-class heatmap.
      
    Additional skip connections are added:
      - From encoder blocks (out3, out2, out1, stem) to both decoders.
      - From the attention decoder blocks (after attn_convX) to the multi-class decoder at matching resolutions.
      
    The soft-argmax of the multi-class heatmaps is used to produce keypoint coordinates.
    The model returns three outputs:
      - Attention heatmap (N, 1, 128, 128, 128)
      - Multi-class heatmap (N, num_keypoints, 128, 128, 128)
      - Keypoint coordinates (N, 3*num_keypoints)
    """
    def __init__(self, num_keypoints=2, pretrained=False, freeze_backbone=False):
        super().__init__()
        self.num_keypoints = num_keypoints

        # ---------------------------
        # 1) Encoder (3D ResNet-18 Backbone)
        # ---------------------------
        backbone = r3d_18(pretrained=pretrained)
        backbone.fc = nn.Identity()  # Remove classification head
        old_conv = backbone.stem[0]
        new_conv = nn.Conv3d(
            in_channels=old_conv.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=(2, 2, 2),  # Downsample depth as well.
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
        # Encoder outputs:
        #   stem_out: (N, 64, 64, 64, 64)   (input resolution: 128^3 -> 64^3)
        #   out1:    (N, 64, 64, 64, 64)
        #   out2:    (N, 128, 32, 32, 32)
        #   out3:    (N, 256, 16, 16, 16)
        #   out4:    (N, 512, 8, 8, 8)

        # ---------------------------
        # 2) Skip Connection Transformations (for encoder to decoder)
        # ---------------------------
        # For multi-class decoder:
        self.skip_conv0 = nn.Conv3d(64, self.num_keypoints, kernel_size=1)   # from stem_out (64^3 resolution)
        self.skip_conv2 = nn.Conv3d(256, 128, kernel_size=1)  # from out3 (16^3 resolution)
        self.skip_conv3 = nn.Conv3d(128, 64, kernel_size=1)   # from out2 (32^3 resolution)
        self.skip_conv4 = nn.Conv3d(64, 32, kernel_size=1)    # from out1 (64^3 resolution)
        
        # For attention decoder:
        self.attn_skip_conv1 = nn.Conv3d(256, 128, kernel_size=1)  # from out3 (16^3 resolution)
        self.attn_skip_conv2 = nn.Conv3d(128, 64, kernel_size=1)   # from out2 (32^3 resolution)
        self.attn_skip_conv3 = nn.Conv3d(64, 32, kernel_size=1)    # from out1 (64^3 resolution)
        self.attn_skip_conv4 = nn.Conv3d(64, self.num_keypoints, kernel_size=1)    # from stem_out (64^3 resolution)

        # ---------------------------
        # 3) Multi-class Decoder (Keypoint Heatmap Branch)
        # ---------------------------
        # Upsampling blocks similar to a UNet decoder.
        self.up1 = nn.ConvTranspose3d(512, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.ConvTranspose3d(32, self.num_keypoints, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(self.num_keypoints),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.num_keypoints, self.num_keypoints, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.num_keypoints),
            nn.ReLU(inplace=True)
        )
        # Final head for multi-class heatmaps.
        self.multi_heatmap_head = nn.Conv3d(self.num_keypoints, num_keypoints, kernel_size=1)

        # ---------------------------
        # 4) Attention Decoder (Single-Class Heatmap Branch)
        # ---------------------------
        self.attn_up1 = nn.ConvTranspose3d(512, 128, kernel_size=2, stride=2)
        self.attn_conv1 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.attn_up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.attn_conv2 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.attn_up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.attn_conv3 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.attn_up4 = nn.ConvTranspose3d(32, self.num_keypoints, kernel_size=2, stride=2)
        self.attn_conv4 = nn.Sequential(
            nn.BatchNorm3d(self.num_keypoints),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.num_keypoints, self.num_keypoints, kernel_size=3, padding=1),
            nn.BatchNorm3d(self.num_keypoints),
            nn.ReLU(inplace=True)
        )
        self.attn_heatmap_head = nn.Conv3d(self.num_keypoints, 1, kernel_size=1)

    def forward(self, x):
        """
        :param x: (N, 1, 128, 128, 128) single-channel volumes.
                  They are replicated to 3 channels for the backbone.
        :return: (attn_heatmap, multi_heatmap, coords)
            - attn_heatmap: (N, 1, 128, 128, 128) single-class attention heatmap.
            - multi_heatmap: (N, num_keypoints, 128, 128, 128) keypoint heatmaps.
            - coords: (N, 3*num_keypoints) predicted keypoints (via soft-argmax on multi_heatmap).
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        # ---------------------------
        # Encoder
        # ---------------------------
        stem_out = self.backbone.stem(x)          # (N, 64, 64, 64, 64)
        out1 = self.backbone.layer1(stem_out)       # (N, 64, 64, 64, 64)
        out2 = self.backbone.layer2(out1)           # (N, 128, 32, 32, 32)
        out3 = self.backbone.layer3(out2)           # (N, 256, 16, 16, 16)
        out4 = self.backbone.layer4(out3)           # (N, 512, 8, 8, 8)

        # ---------------------------
        # Attention Decoder (Single-Class Heatmap Branch)
        # ---------------------------
        attn1 = self.attn_up1(out4)               # (N, 128, 16, 16, 16)
        attn1 = self.attn_conv1(attn1)
        # Add skip connection from encoder out3
        attn1 = attn1 + self.attn_skip_conv1(out3)

        attn2 = self.attn_up2(attn1)               # (N, 64, 32, 32, 32)
        attn2 = self.attn_conv2(attn2)
        attn2 = attn2 + self.attn_skip_conv2(out2)

        attn3 = self.attn_up3(attn2)               # (N, 32, 64, 64, 64)
        attn3 = self.attn_conv3(attn3)
        attn3 = attn3 + self.attn_skip_conv3(out1)

        attn4 = self.attn_up4(attn3)               # (N, num_keypoints, 128, 128, 128)
        attn4 = self.attn_conv4(attn4)
        attn4 = attn4 + F.interpolate(self.attn_skip_conv4(stem_out),
                                      size=attn4.shape[2:], mode='trilinear', align_corners=False)
        attn_heatmap = self.attn_heatmap_head(attn4)  # (N, 1, 128, 128, 128)

        # ---------------------------
        # Multi-class Decoder (Keypoint Heatmap Branch)
        # ---------------------------
        up1 = self.up1(out4)                      # (N, 128, 16, 16, 16)
        up1 = self.conv1(up1)
        # Add skip connection from encoder out3 and attention branch (attn1)
        up1 = up1 + self.skip_conv2(out3) + attn1

        up2 = self.up2(up1)                       # (N, 64, 32, 32, 32)
        up2 = self.conv2(up2)
        up2 = up2 + self.skip_conv3(out2) + attn2

        up3 = self.up3(up2)                       # (N, 32, 64, 64, 64)
        up3 = self.conv3(up3)
        up3 = up3 + self.skip_conv4(out1) + attn3

        up4 = self.up4(up3)                       # (N, num_keypoints, 128, 128, 128)
        up4 = self.conv4(up4)
        # Add skip connection from stem and attention branch (attn4)
        skip0_up = F.interpolate(self.skip_conv0(stem_out),
                                 size=up4.shape[2:], mode='trilinear', align_corners=False)
        up4 = up4 + skip0_up + attn4

        multi_heatmap = self.multi_heatmap_head(up4)  # (N, num_keypoints, 128, 128, 128)

        # ---------------------------
        # Keypoint Coordinates from Multi-class Heatmap
        # ---------------------------
        coords = soft_argmax_3d(multi_heatmap)     # (N, 3*num_keypoints)

        return attn_heatmap, multi_heatmap, coords
