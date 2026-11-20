"""BrainSignsNET (revised model): 3D ResNet-18 encoder, attention decoder branch and multi-class heatmap decoder,
landmark coordinates from a 3D soft-argmax over the multi-class heatmaps."""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models.video import r3d_18

LANDMARKS = ["CCPost", "CCAnt", "RPutamen", "LPutamen", "LCaudate", "LThalamus", "LPallidum", "LHippocampus", "LAmygdala",
             "RCaudate", "RThalamus", "RPallidum", "RHippocampus", "RAmygdala", "OpticChiasm", "IntraventricularForamen",
             "Midbrain", "Pons", "Medulla", "LInternalCapsulGenu", "RInternalCapsulGenu"]


def soft_argmax_3d(heatmaps):
    N, K, D, H, W = heatmaps.shape
    sm = F.softmax(heatmaps.view(N, K, -1), dim=2)
    dev = heatmaps.device
    zz, yy, xx = torch.meshgrid(torch.linspace(0, D - 1, D, device=dev), torch.linspace(0, H - 1, H, device=dev),
                                torch.linspace(0, W - 1, W, device=dev), indexing="ij")
    zz, yy, xx = zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)
    return torch.stack([(sm * zz).sum(2), (sm * yy).sum(2), (sm * xx).sum(2)], dim=2)


def _blk(c):
    return nn.Sequential(nn.BatchNorm3d(c), nn.ReLU(inplace=True), nn.Conv3d(c, c, 3, padding=1),
                         nn.BatchNorm3d(c), nn.ReLU(inplace=True))


class BrainSignsNET(nn.Module):
    def __init__(self, num_keypoints=len(LANDMARKS)):
        super().__init__()
        backbone = r3d_18(weights=None); backbone.fc = nn.Identity()
        old = backbone.stem[0]
        backbone.stem[0] = nn.Conv3d(old.in_channels, old.out_channels, kernel_size=old.kernel_size, stride=(2, 2, 2),
                                     padding=old.padding, bias=False)
        self.backbone = backbone
        K = num_keypoints
        self.skip_conv0 = nn.Conv3d(64, K, 1)
        self.skip_conv2 = nn.Conv3d(256, 128, 1)
        self.skip_conv3 = nn.Conv3d(128, 64, 1)
        self.skip_conv4 = nn.Conv3d(64, 32, 1)
        self.up1, self.conv1 = nn.ConvTranspose3d(512, 128, 2, 2), _blk(128)
        self.up2, self.conv2 = nn.ConvTranspose3d(128, 64, 2, 2), _blk(64)
        self.up3, self.conv3 = nn.ConvTranspose3d(64, 32, 2, 2), _blk(32)
        self.up4, self.conv4 = nn.ConvTranspose3d(32, K, 2, 2), _blk(K)
        self.multi_heatmap_head = nn.Conv3d(K, K, 1)
        self.attn_skip_conv1 = nn.Conv3d(256, 128, 1)
        self.attn_skip_conv2 = nn.Conv3d(128, 64, 1)
        self.attn_skip_conv3 = nn.Conv3d(64, 32, 1)
        self.attn_skip_conv4 = nn.Conv3d(64, K, 1)
        self.attn_up1, self.attn_conv1 = nn.ConvTranspose3d(512, 128, 2, 2), _blk(128)
        self.attn_up2, self.attn_conv2 = nn.ConvTranspose3d(128, 64, 2, 2), _blk(64)
        self.attn_up3, self.attn_conv3 = nn.ConvTranspose3d(64, 32, 2, 2), _blk(32)
        self.attn_up4, self.attn_conv4 = nn.ConvTranspose3d(32, K, 2, 2), _blk(K)
        self.attn_heatmap_head = nn.Conv3d(K, 1, 1)

    def forward(self, x):
        """x: (N, 1, 128, 128, 128). Returns attention heatmap, multi-class heatmaps and coordinates (N, K, 3) on the grid."""
        x = x.repeat(1, 3, 1, 1, 1) if x.size(1) == 1 else x
        stem = self.backbone.stem(x)
        o1 = self.backbone.layer1(stem); o2 = self.backbone.layer2(o1)
        o3 = self.backbone.layer3(o2); o4 = self.backbone.layer4(o3)
        a1 = self.attn_conv1(self.attn_up1(o4)) + self.attn_skip_conv1(o3)
        a2 = self.attn_conv2(self.attn_up2(a1)) + self.attn_skip_conv2(o2)
        a3 = self.attn_conv3(self.attn_up3(a2)) + self.attn_skip_conv3(o1)
        a4 = self.attn_conv4(self.attn_up4(a3))
        a4 = a4 + F.interpolate(self.attn_skip_conv4(stem), size=a4.shape[2:], mode="trilinear", align_corners=False)
        attn = self.attn_heatmap_head(a4)
        u1 = self.conv1(self.up1(o4)) + self.skip_conv2(o3) + a1
        u2 = self.conv2(self.up2(u1)) + self.skip_conv3(o2) + a2
        u3 = self.conv3(self.up3(u2)) + self.skip_conv4(o1) + a3
        u4 = self.conv4(self.up4(u3))
        u4 = u4 + F.interpolate(self.skip_conv0(stem), size=u4.shape[2:], mode="trilinear", align_corners=False) + a4
        heat = self.multi_heatmap_head(u4)
        return attn, heat, soft_argmax_3d(heat)


def load(weights, device="cpu"):
    m = BrainSignsNET()
    sd = torch.load(weights, map_location="cpu")
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    return m.to(device).eval()
