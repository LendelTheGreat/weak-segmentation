import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, logger):
        super(UNet, self).__init__()
        self.logger = logger
        
        self.down0 = ConvBlock(3, 64)
        self.down1 = ConvBlock(64, 128)
        self.down2 = ConvBlock(128, 256)
        self.down3 = ConvBlock(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        
        self.bottom = ConvBlock(512, 1024)

        self.upsample3 = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                                       nn.Conv2d(1024, 512, kernel_size=1))
        self.up3 = ConvBlock(1024, 512)
        self.upsample2 = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                                       nn.Conv2d(512, 256, kernel_size=1))
        self.up2 = ConvBlock(512, 256)
        self.upsample1 = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                                       nn.Conv2d(256, 128, kernel_size=1))
        self.up1 = ConvBlock(256, 128)
        self.upsample0 = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                                       nn.Conv2d(128, 64, kernel_size=1))
        self.up0 = ConvBlock(128, 64)
        
        self.class_conv = nn.Conv2d(64, 5, kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        down0 = self.down0(x)
        x = self.maxpool(down0)
        down1 = self.down1(x)
        x = self.maxpool(down1)
        down2 = self.down2(x)
        x = self.maxpool(down2)
        down3 = self.down3(x)
        x = self.maxpool(down3)
        
        x = self.bottom(x)
        
        x = self.upsample3(x)
        x = torch.cat([x, down3], dim=1)   
        x = self.up3(x)
        
        x = self.upsample2(x)
        x = torch.cat([x, down2], dim=1)   
        x = self.up2(x)
        
        x = self.upsample1(x)
        x = torch.cat([x, down1], dim=1)   
        x = self.up1(x)
        
        x = self.upsample0(x)
        x = torch.cat([x, down0], dim=1)   
        x = self.up0(x)
        
        x = self.class_conv(x)
        
        x = self.softmax(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)) 
    def forward(self, x):
        return self.conv(x)