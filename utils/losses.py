import torch
import torch.nn as nn
import torch.nn.functional as F

# Fonctions de perte pour l'entraînement
class DetectionLoss(nn.Module):
    def __init__(self, num_classes, cls_weight=1.0, reg_weight=1.0):
        """
        Initialise les fonctions de perte pour la détection d'objets
        
        Args:
            num_classes (int): Nombre de classes (incluant le fond)
            cls_weight (float): Poids de la perte de classification
            reg_weight (float): Poids de la perte de régression
        """
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.classification_loss = nn.CrossEntropyLoss(reduction='none')
        self.regression_loss = nn.SmoothL1Loss(reduction='none', beta=0.11)
        
    def forward(self, cls_preds, reg_preds, cls_targets, reg_targets, pos_mask):
        """
        Calcule la perte totale pour la détection d'objets
        
        Args:
            cls_preds: Prédictions de classification [batch_size, num_anchors, num_classes]
            reg_preds: Prédictions de régression [batch_size, num_anchors, 4]
            cls_targets: Cibles de classification [batch_size, num_anchors]
            reg_targets: Cibles de régression [batch_size, num_anchors, 4]
            pos_mask: Masque des anchors positifs [batch_size, num_anchors]
            
        Returns:
            tuple: (total_loss, cls_loss, reg_loss)
        """
        batch_size = cls_preds.size(0)
        
        # Calculer la perte de classification
        cls_loss = self.classification_loss(
            cls_preds.view(-1, self.num_classes),
            cls_targets.view(-1)
        )
        cls_loss = cls_loss.view(batch_size, -1)
        
        # Appliquer la pondération pour les échantillons positifs/négatifs
        # Utiliser tous les échantillons pour la classification
        cls_loss = cls_loss.mean()
        
        # Calculer la perte de régression (seulement pour les anchors positifs)
        reg_loss = self.regression_loss(
            reg_preds,
            reg_targets
        )
        reg_loss = reg_loss.sum(dim=2)  # Somme sur les coordonnées x, y, w, h
        
        # Appliquer le masque des anchors positifs
        pos_count = pos_mask.sum().float().clamp(min=1.0)  # Éviter la division par zéro
        reg_loss = (reg_loss * pos_mask).sum() / pos_count
        
        # Perte totale = perte de classification + perte de régression
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return total_loss, cls_loss, reg_loss