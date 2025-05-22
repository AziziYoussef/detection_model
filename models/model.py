import torch
import torch.nn as nn

from .backbone import Backbone
from .fpn import FeaturePyramidNetwork
from .prediction import PredictionHeads

# Modèle complet de détection d'objets
class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=False):
        """
        Initialise le modèle de détection d'objets
        
        Args:
            num_classes (int): Nombre de classes à détecter (sans compter le fond)
            pretrained_backbone (bool): Si True, utilise un backbone pré-entraîné
        """
        super(ObjectDetectionModel, self).__init__()
        
        # Nombre de classes (inclut le fond)
        self.num_classes = num_classes + 1  # +1 pour le fond
        
        # Backbone: extraction de caractéristiques
        self.backbone = Backbone()
        
        # Dimensions des caractéristiques à chaque niveau du backbone
        in_channels_list = [64, 128, 256, 512, 512]
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(in_channels_list, 256)
        
        # Têtes de prédiction pour chaque niveau de la pyramide
        self.prediction_heads = nn.ModuleList([
            PredictionHeads(256, self.num_classes) for _ in range(len(in_channels_list))
        ])
        
        # Initialisation des poids
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialise les poids du modèle avec Kaiming normal"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass du modèle
        
        Args:
            x (torch.Tensor): Batch d'images [batch_size, 3, height, width]
            
        Returns:
            tuple: (cls_preds, reg_preds) où:
                  cls_preds: Prédictions de classification [batch_size, num_anchors, num_classes]
                  reg_preds: Prédictions de régression [batch_size, num_anchors, 4]
        """
        # Extraction des caractéristiques via le backbone
        features = self.backbone(x)
        
        # Application du FPN
        fpn_features = self.fpn(features)
        
        # Prédictions à chaque niveau
        all_cls_preds = []
        all_reg_preds = []
        
        for feature, head in zip(fpn_features, self.prediction_heads):
            cls_pred, reg_pred = head(feature)
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
        
        # Concaténer toutes les prédictions
        cls_preds = torch.cat(all_cls_preds, dim=1)
        reg_preds = torch.cat(all_reg_preds, dim=1)
        
        return cls_preds, reg_preds