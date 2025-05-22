import torch

def box_iou(boxes1, boxes2):
    """
    Calcule l'IoU entre deux ensembles de boîtes
    
    Args:
        boxes1 (torch.Tensor): Premier ensemble de boîtes, format [N, 4]
        boxes2 (torch.Tensor): Deuxième ensemble de boîtes, format [M, 4]
        
    Returns:
        torch.Tensor: IoU de forme [N, M]
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]
    
    # Calcul de l'intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Calcul de l'union
    union = area1[:, None] + area2 - intersection
    
    iou = intersection / union  # [N, M]
    
    return iou

def box_area(boxes):
    """
    Calcule l'aire de chaque boîte
    
    Args:
        boxes (torch.Tensor): Boîtes de format [N, 4], où chaque boîte est [x1, y1, x2, y2]
        
    Returns:
        torch.Tensor: Aires de forme [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def assign_targets_to_anchors(gt_boxes, gt_labels, anchors, pos_iou_thresh=0.5, neg_iou_thresh=0.4):
    """
    Assigne des cibles aux anchors
    
    Args:
        gt_boxes (list): Liste de tenseurs de boîtes ground truth pour chaque image
        gt_labels (list): Liste de tenseurs d'étiquettes ground truth pour chaque image
        anchors (torch.Tensor): Anchors de forme [num_anchors, 4]
        pos_iou_thresh (float): Seuil IoU pour les anchors positifs
        neg_iou_thresh (float): Seuil IoU pour les anchors négatifs
        
    Returns:
        tuple: (cls_targets, reg_targets, pos_mask)
    """
    batch_size = len(gt_boxes)
    num_anchors = anchors.shape[0]
    device = anchors.device
    
    # Initialiser les cibles
    cls_targets = torch.zeros((batch_size, num_anchors), dtype=torch.long, device=device)
    reg_targets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float, device=device)
    pos_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=device)
    
    # Pour chaque image dans le batch
    for i in range(batch_size):
        if len(gt_boxes[i]) == 0:
            continue
            
        # Calculer IoU entre anchors et boîtes ground truth
        ious = box_iou(anchors, gt_boxes[i])  # [num_anchors, num_gt]
        
        # Pour chaque anchor, trouver la meilleure boîte ground truth
        max_iou, max_idx = ious.max(dim=1)
        
        # Anchors positifs: IoU > pos_iou_thresh
        pos_mask[i] = max_iou > pos_iou_thresh
        
        # Assigner les étiquettes et les cibles de régression
        cls_targets[i][pos_mask[i]] = gt_labels[i][max_idx[pos_mask[i]]]
        
        # Calculer les cibles de régression pour les anchors positifs
        matched_gt_boxes = gt_boxes[i][max_idx[pos_mask[i]]]
        pos_anchors = anchors[pos_mask[i]]
        
        # Encoder les cibles de régression (utilisant la transformation x, y, w, h)
        reg_targets[i][pos_mask[i]] = encode_boxes(matched_gt_boxes, pos_anchors)
    
    return cls_targets, reg_targets, pos_mask

def encode_boxes(gt_boxes, anchors):
    """
    Encode les boîtes ground truth en cibles de régression
    
    Args:
        gt_boxes (torch.Tensor): Boîtes ground truth de forme [N, 4]
        anchors (torch.Tensor): Anchors de forme [N, 4]
        
    Returns:
        torch.Tensor: Cibles de régression de forme [N, 4]
    """
    # Convertir les boîtes [x1, y1, x2, y2] en [cx, cy, w, h]
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    
    # Encoder les cibles
    tx = (gt_cx - anchor_cx) / anchor_w
    ty = (gt_cy - anchor_cy) / anchor_h
    tw = torch.log(gt_w / anchor_w)
    th = torch.log(gt_h / anchor_h)
    
    return torch.stack([tx, ty, tw, th], dim=1)

def decode_boxes(reg_preds, anchors):
    """
    Décode les prédictions de régression en boîtes
    
    Args:
        reg_preds (torch.Tensor): Prédictions de régression de forme [batch_size, num_anchors, 4]
        anchors (torch.Tensor): Anchors de forme [num_anchors, 4]
        
    Returns:
        torch.Tensor: Boîtes décodées de forme [batch_size, num_anchors, 4]
    """
    # Extraire les composantes des anchors
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    
    # Decoder les prédictions
    tx = reg_preds[..., 0]
    ty = reg_preds[..., 1]
    tw = reg_preds[..., 2]
    th = reg_preds[..., 3]
    
    cx = tx * anchor_w.unsqueeze(0) + anchor_cx.unsqueeze(0)
    cy = ty * anchor_h.unsqueeze(0) + anchor_cy.unsqueeze(0)
    w = torch.exp(tw) * anchor_w.unsqueeze(0)
    h = torch.exp(th) * anchor_h.unsqueeze(0)
    
    boxes = torch.stack([
        cx - w/2,  # x1
        cy - h/2,  # y1
        cx + w/2,  # x2
        cy + h/2   # y2
    ], dim=-1)
    
    return boxes

def nms(boxes, scores, iou_threshold=0.5):
    """
    Applique le Non-Maximum Suppression
    
    Args:
        boxes (torch.Tensor): Boîtes de forme [N, 4]
        scores (torch.Tensor): Scores de forme [N]
        iou_threshold (float): Seuil IoU pour la suppression
        
    Returns:
        torch.Tensor: Indices des boîtes conservées
    """
    # Vérification de sécurité: si boxes est vide, retourner un tenseur vide
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
        
    # Trier les boîtes par score
    _, order = scores.sort(descending=True)
    
    # Vérification de sécurité: si order est vide, retourner un tenseur vide
    if order.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    
    keep = []
    while order.numel() > 0:
        # Vérification de sécurité supplémentaire
        if order.numel() == 1:
            keep.append(order.item())
            break
            
        i = order[0].item()
        keep.append(i)
        
        # Calculer IoU avec les boîtes restantes
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])
        
        # Vérification de sécurité: si ious est vide
        if ious.numel() == 0:
            break
            
        # Garder les boîtes avec IoU < threshold
        inds = torch.where(ious[0] <= iou_threshold)[0]
        
        # Vérification de sécurité: si inds est vide
        if inds.numel() == 0:
            break
            
        order = order[1:][inds + 1]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)