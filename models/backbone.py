import torch
import torch.nn as nn

# Définition d'un bloc convolutionnel avec activation et normalisation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            stride=stride, padding=kernel_size//2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        # Pas d'inplace pour éviter les problèmes avec la précision mixte
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Définition d'un bloc résiduel sans operations inplace
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        # Utilisation de l'addition sans inplace
        x = x + residual  # NON-inplace (pas de +=)
        return x

# Backbone: Extraction de caractéristiques à plusieurs échelles
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        
        # Couche d'entrée
        self.input_conv = ConvBlock(3, 64, kernel_size=7, stride=2)  # 1/2
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1/4
        
        # Blocs de caractéristiques à différentes échelles
        self.layer1 = nn.Sequential(
            ConvBlock(64, 64),
            ResidualBlock(64),
            ConvBlock(64, 128, stride=2)  # 1/8
        )
        
        self.layer2 = nn.Sequential(
            ConvBlock(128, 128),
            ResidualBlock(128),
            ConvBlock(128, 256, stride=2)  # 1/16
        )
        
        self.layer3 = nn.Sequential(
            ConvBlock(256, 256),
            ResidualBlock(256),
            ConvBlock(256, 512, stride=2)  # 1/32
        )
        
        self.layer4 = nn.Sequential(
            ConvBlock(512, 512),
            ResidualBlock(512)
        )
        
    def forward(self, x):
        # Collecter les caractéristiques à différentes échelles
        features = []
        
        x = self.input_conv(x)
        x = self.pool1(x)
        features.append(x)  # C1: 1/4
        
        x = self.layer1(x)
        features.append(x)  # C2: 1/8
        
        x = self.layer2(x)
        features.append(x)  # C3: 1/16
        
        x = self.layer3(x)
        features.append(x)  # C4: 1/32
        
        x = self.layer4(x)
        features.append(x)  # C5: 1/32 avec plus de contexte
        
        return features