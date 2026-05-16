
import torch.nn as nn
import torch
class HR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HR, self).__init__()

        # Left
        self.conv3x1_e1 = nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0))
        self.conv1x3_e1 = nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 1))
        self.relu_e1 = nn.ReLU(inplace=True)
        self.bn_e1 = nn.BatchNorm2d(out_ch)

        self.pool_e1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3x1_e2 = nn.Conv2d(out_ch, out_ch, (3, 1), padding=(1, 0))
        self.conv1x3_e2 = nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 1))
        self.relu_e2 = nn.ReLU(inplace=True)
        self.bn_e2 = nn.BatchNorm2d(out_ch)

        self.pool_e2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3x1_e3 = nn.Conv2d(out_ch, out_ch, (3, 1), padding=(2, 0), dilation=(2, 1))
        self.conv1x3_e3 = nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 2), dilation=(1, 2))
        self.relu_e3 = nn.ReLU(inplace=True)
        self.bn_e3 = nn.BatchNorm2d(out_ch)

        self.pool_e3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3x1_e4 = nn.Conv2d(out_ch, out_ch, (3, 1), padding=(4, 0), dilation=(4, 1))
        self.conv1x3_e4 = nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 4), dilation=(1, 4))
        self.relu_e4 = nn.ReLU(inplace=True)
        self.bn_e4 = nn.BatchNorm2d(out_ch)

        self.pool_e4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Bridge
        self.convb = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bnb = nn.BatchNorm2d(out_ch)
        self.relub = nn.ReLU(inplace=True)

        # Right
        self.upsample_d4 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3x1_d4 = nn.Conv2d(out_ch, out_ch, (3, 1), padding=(4, 0), dilation=(4, 1))
        self.conv1x3_d4 = nn.Conv2d(out_ch*2, out_ch, (1, 3), padding=(0, 4), dilation=(1, 4))
        self.relu_d4 = nn.ReLU(inplace=True)
        self.bn_d4 = nn.BatchNorm2d(out_ch)

        self.upsample_d3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3x1_d3 = nn.Conv2d(out_ch, out_ch, (3, 1), padding=(2, 0), dilation=(2, 1))
        self.conv1x3_d3 = nn.Conv2d(out_ch*2, out_ch, (1, 3), padding=(0, 2), dilation=(1, 2))
        self.relu_d3 = nn.ReLU(inplace=True)
        self.bn_d3 = nn.BatchNorm2d(out_ch)

        self.upsample_d2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3x1_d2 = nn.Conv2d(out_ch, out_ch, (3, 1), padding=(1, 0))
        self.conv1x3_d2 = nn.Conv2d(out_ch*2, out_ch, (1, 3), padding=(0, 1))
        self.relu_d2 = nn.ReLU(inplace=True)
        self.bn_d2 = nn.BatchNorm2d(out_ch)

        self.upsample_d1 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv3x1_d1 = nn.Conv2d(out_ch, out_ch, (3, 1), padding=(1, 0))
        self.conv1x3_d1 = nn.Conv2d((out_ch+out_ch), out_ch, (1, 3), padding=(0, 1))
        self.relu_d1 = nn.ReLU(inplace=True)
        self.bn_d1 = nn.BatchNorm2d(out_ch)


    def forward(self, x):
        hx = x                   # identity connections

        ex1 = self.relu_e1(self.conv3x1_e1(hx))
        hx = self.relu_e1(self.bn_e1(self.conv1x3_e1(ex1)))
        hx = self.pool_e1(hx)

        ex2 = self.relu_e2(self.conv3x1_e2(hx))
        hx = self.relu_e2(self.bn_e2(self.conv1x3_e2(ex2)))
        hx = self.pool_e2(hx)

        ex3 = self.relu_e3(self.conv3x1_e3(hx))
        hx = self.relu_e3(self.bn_e3(self.conv1x3_e3(ex3)))
        hx = self.pool_e3(hx)

        ex4 = self.relu_e4(self.conv3x1_e4(hx))
        hx = self.relu_e4(self.bn_e4(self.conv1x3_e4(ex4)))
        hx = self.pool_e4(hx)

        eb = self.relub(self.bnb(self.convb(hx)))

        hx = self.upsample_d4(eb)
        hx = self.relu_d4(self.conv3x1_d4(hx))
        dx4 = self.relu_d4(self.bn_d4(self.conv1x3_d4(torch.cat((hx, ex4), 1))))

        hx = self.upsample_d3(dx4)
        hx = self.relu_d3(self.conv3x1_d3(hx))
        dx3 = self.relu_d3(self.bn_d3(self.conv1x3_d3(torch.cat((hx, ex3), 1))))

        hx = self.upsample_d2(dx3)
        hx = self.relu_d2(self.conv3x1_d2(hx))
        dx2 = self.relu_d2(self.bn_d2(self.conv1x3_d2(torch.cat((hx, ex2), 1))))

        hx = self.upsample_d1(dx2)
        hx = self.relu_d1(self.conv3x1_d1(hx))
        dx1 = self.relu_d1(self.bn_d1(self.conv1x3_d1(torch.cat((hx, ex1), 1))))

        return x + dx1
