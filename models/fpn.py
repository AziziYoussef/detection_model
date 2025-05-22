import torch
import torch.nn as nn
import torch.nn.functional as F

# Network d'architecture en pyramide (FPN)
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        
        # Couches latérales (réduisent les dimensions des canaux)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Couches de lissage après l'addition
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])
        
    def forward(self, features):
        # Dernière caractéristique (la plus profonde)
        last_feature = self.lateral_convs[-1](features[-1])
        
        # Initialiser la liste des caractéristiques de sortie avec la dernière
        fpn_features = [self.smooth_convs[-1](last_feature)]
        
        # Construire la pyramide de bas en haut
        for i in range(len(features) - 2, -1, -1):
            # Convertir les dimensions des canaux
            lateral = self.lateral_convs[i](features[i])
            
            # Redimensionner la caractéristique de niveau supérieur et ajouter
            top_down = F.interpolate(fpn_features[0], size=lateral.shape[2:], mode='nearest')
            fpn_feature = lateral + top_down  # NON-inplace addition
            
            # Appliquer un lissage et ajouter au début de la liste
            fpn_features.insert(0, self.smooth_convs[i](fpn_feature))
        
        return fpn_features