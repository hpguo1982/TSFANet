
import torch.nn as nn
import torch
class MaxAvg(nn.Module):
    def __init__(self):
        super(MaxAvg, self).__init__()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return x
class HighFre(nn.Module):
    def __init__(self,out_channels):
        super(HighFre,self).__init__()
        self.freq_proj = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=True)  # 输出单通道的频域响应
        )
    def forward(self,feat):
        # feat: B,C,H,W
        F = torch.fft.fft2(feat, norm='ortho')    # complex B,C,H,W
        Fr = torch.view_as_real(F)                # B,C,H,W,2 -> real representation
            # reshape to B,2C,H,W for conv
        Fr = Fr.permute(0,1,4,2,3).reshape(feat.size(0), feat.size(1)*2, feat.size(2), feat.size(3))
        m = self.freq_proj(Fr)  # B,1,H,W
        return m
class CLAI(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CLAI, self).__init__()
        self.conv_block3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_block5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv_block7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.MaxAvg = MaxAvg()
        self.SharedConvLayer = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu =nn.ReLU()
        self.highFre1 = HighFre(out_channels=out_channels)
        self.highFre2 = HighFre(out_channels=out_channels)
        self.highFre3 = HighFre(out_channels=out_channels)
        self.edge_branch = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.spatial_norm = nn.Softmax(dim=1)
    def forward(self, x):
        identity = x

        conv_3x3 = self.relu(self.conv_block3x3(x))
        x_ = self.MaxAvg(conv_3x3)
        x_ = self.SharedConvLayer(x_)
        conv_3x3_sigmoid = self.sigmoid(x_)
        conv_3x3 = conv_3x3 * conv_3x3_sigmoid

        conv_5x5 = self.relu(self.conv_block5x5(x))
        x_ = self.MaxAvg(conv_5x5)
        x_ = self.SharedConvLayer(x_)
        conv_5x5_sigmoid = self.sigmoid(x_)
        conv_5x5 = conv_5x5 * conv_5x5_sigmoid


        conv_7x7 = self.relu(self.conv_block7x7(x))
        x_ = self.MaxAvg(conv_7x7)
        x_ = self.SharedConvLayer(x_)
        conv_7x7_sigmoid = self.sigmoid(x_)
        conv_7x7 = conv_7x7 * conv_7x7_sigmoid


        m3 = self.highFre1(conv_3x3)
        m7 = self.highFre2(conv_5x5)
        m11 = self.highFre3(conv_7x7)

        ms = torch.cat([m3, m7, m11], dim=1)  # B,3,H,W
        ws = self.spatial_norm(ms)            # softmax in scale dim

        f3_out = conv_3x3 * ws[:,0:1,:,:]
        f7_out = conv_5x5 * ws[:,1:2,:,:]
        f11_out = conv_7x7 * ws[:,2:3,:,:]
        fe = self.relu(self.edge_branch(x))
        out = f3_out + f7_out + f11_out + 0.15 * fe
        return out + identity
