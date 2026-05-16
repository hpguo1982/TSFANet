import torch
import torch.nn as nn
import torch.nn.functional as F

class IEFI(nn.Module):

    def __init__(self, in_channels, mid_channels=None):
        super(IEFI, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels

        self.channel_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv2d(mid_channels + in_channels, in_channels, kernel_size=1)

    def forward(self, feat1, feat2):

        if feat1.shape[2:] != feat2.shape[2:]:
            feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)


        cat = torch.cat([feat1, feat2], dim=1)
        ch_out = self.channel_fuse(cat)


        sp_out = self.spatial_fuse(feat1 + feat2)


        fused = torch.cat([ch_out, sp_out], dim=1)
        out = self.out_conv(fused)
        return out
