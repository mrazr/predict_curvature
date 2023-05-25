import torch
from torch import nn
import torch.nn.functional as F


class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=F.leaky_relu):
        super().__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same')
        self.bn = nn.BatchNorm2d(out_channels)

        self.act = activation

    def forward(self, x):
        x = self.act(self.conv1(x))
        return self.act(self.bn(self.conv2(x)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = UNetBlock(1, out_channels=8)
        self.block2 = UNetBlock(8, out_channels=32)
        self.block3 = UNetBlock(32, out_channels=64)
        self.block4 = UNetBlock(64, out_channels=64)

        self._intermediate = []

    def forward(self, x):
        x1 = self.block1(x)
        x = F.max_pool2d(x1, 2, 2)
        x2 = self.block2(x)
        x = F.max_pool2d(x2, 2, 2)
        x3 = self.block3(x)
        x = F.max_pool2d(x3, 2, 2)
        x4 = self.block4(x)

        self._intermediate = [x1, x2, x3]

        return x4


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding='same')
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = F.leaky_relu(self.conv(x))

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = UNetBlock(128, 64)
        self.block2 = UNetBlock(96, 32)
        self.block3 = UNetBlock(40, 1, activation=F.tanh)

        self.upsample1 = UpsampleBlock(64, 64)
        self.upsample2 = UpsampleBlock(64, 64)
        self.upsample3 = UpsampleBlock(32, 32)

    def forward(self, x, xs):
        x = self.upsample1(x)
        x = torch.cat((x, xs[2]), dim=1)
        x = F.leaky_relu(self.block1(x))

        x = self.upsample2(x)
        x = torch.cat((x, xs[1]), dim=1)
        x = F.leaky_relu(self.block2(x))

        x = self.upsample3(x)
        x = torch.cat((x, xs[0]), dim=1)
        x = self.block3(x)

        return F.tanh(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm2d(1)
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        x = self.bn(x)
        x = self.encoder(x)

        return self.decoder(x, self.encoder._intermediate)


