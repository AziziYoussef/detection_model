import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pycocotools.coco import COCO

class LostObjectsDataset(Dataset):
    """Dataset personnalisé pour la détection d'objets perdus"""
    
    def __init__(self, coco, img_ids, img_dir, class_ids, transform=None):
        """
        Initialise le dataset
        
        Args:
            coco (COCO): Instance COCO avec les annotations
            img_ids (list): Liste des IDs d'images à utiliser
            img_dir (str): Chemin vers le dossier contenant les images
            class_ids (list): Liste des IDs de classes à utiliser
            transform (callable, optional): Transformations à appliquer
        """
        self.coco = coco
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.class_ids = class_ids
        self.transform = transform
        
        # Mapping de catégories COCO vers nos indices
        self.cat_mapping = {cat_id: i + 1 for i, cat_id in enumerate(class_ids)}  # +1 car 0 est réservé au fond
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        # Obtenir l'ID de l'image
        img_id = self.img_ids[idx]
        
        # Charger les informations de l'image
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Charger l'image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erreur: impossible de charger l'image {img_path}")
            # Une solution de secours - créer une image noire
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Dimensions de l'image
        height, width = img.shape[:2]
        
        # Récupérer les annotations
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.class_ids)
        anns = self.coco.loadAnns(ann_ids)
        
        # Préparer les boîtes et classes
        boxes = []
        labels = []
        
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in self.class_ids:
                # Convertir le bbox [x, y, width, height] en [x1, y1, x2, y2]
                bbox = ann['bbox']
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                # S'assurer que les coordonnées sont dans les limites de l'image
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                # Vérifier que la boîte est valide (largeur et hauteur > 0)
                if x2 > x1 and y2 > y1:
                    # Normaliser les coordonnées entre 0 et 1
                    x1 /= width
                    y1 /= height
                    x2 /= width
                    y2 /= height
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.cat_mapping[cat_id])
        
        # Convertir en tenseurs
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Si aucune boîte valide, créer des tenseurs vides
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0), dtype=torch.int64)
        
        # Identifiant de l'image pour référence
        image_id = torch.tensor([img_id])
        
        # Surface des boîtes
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0), dtype=torch.float32)
        
        # Supposons que tous les objets sont visibles (pas d'occultation)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Créer le dictionnaire cible
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        
        # Appliquer les transformations
        if self.transform:
            img = self.transform(img)
        
        return img, target