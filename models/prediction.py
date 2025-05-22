import torch
import torch.nn as nn

# Têtes de prédiction pour la classification et la régression
class PredictionHeads(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=9):
        super(PredictionHeads, self).__init__()
        
        # Tête de classification (pour chaque ancre à chaque position)
        self.cls_head = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )
        
        # Tête de régression (4 valeurs: x, y, w, h pour chaque ancre)
        self.reg_head = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, padding=1
        )
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Prédictions de classification
        cls_pred = self.cls_head(x)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
        cls_pred = cls_pred.view(batch_size, -1, self.num_classes)
        
        # Prédictions de régression
        reg_pred = self.reg_head(x)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous()
        reg_pred = reg_pred.view(batch_size, -1, 4)
        
        return cls_pred, reg_pred