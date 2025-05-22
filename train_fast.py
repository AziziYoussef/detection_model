import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torchvision.models.detection as detection_models
from torch.amp import GradScaler, autocast
import time
import random
import cv2
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Optimisations CUDA
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration rapide corrigée (paramètres réduits pour éviter les erreurs de mémoire)
config = {
    'num_classes': 10,
    'batch_size': 4,  # Réduit pour éviter les erreurs de mémoire
    'learning_rate': 0.005,
    'num_epochs': 10,
    'image_size': (320, 320),
    'use_mixed_precision': True,
    'max_train_images': 2000,  # Encore plus réduit
    'max_val_images': 200,     # Réduit
    'optimizer': 'sgd',
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'num_workers': 0,  # Désactiver le multiprocessing pour éviter les erreurs de mémoire
    'pin_memory': False,  # Désactiver pour éviter les erreurs de mémoire
    'coco_dir': 'c:/Users/ay855/Documents/detction_model/coco',
    'output_dir': 'output_fast',
    'classes': [
        'backpack', 'suitcase', 'handbag', 'cell phone', 'laptop',
        'book', 'umbrella', 'bottle', 'keyboard', 'remote'
    ]
}

class LostObjectsDatasetForPretrained(torch.utils.data.Dataset):
    """Dataset adapté pour les modèles pré-entraînés de torchvision"""
    
    def __init__(self, coco, img_ids, img_dir, class_ids, transform=None):
        self.coco = coco
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.class_ids = class_ids
        self.transform = transform
        
        # Mapping de catégories COCO vers nos indices (1-indexed, 0 réservé au fond)
        self.cat_mapping = {cat_id: i + 1 for i, cat_id in enumerate(class_ids)}
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        try:
            # Charger les informations de l'image
            img_info = self.coco.loadImgs([img_id])[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Charger l'image
            img = cv2.imread(img_path)
            if img is None:
                # Image de secours si le chargement échoue
                img = np.zeros((320, 320, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionner l'image pour économiser la mémoire
            img = cv2.resize(img, config['image_size'])
            height, width = img.shape[:2]
            
            # Récupérer les annotations
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.class_ids)
            anns = self.coco.loadAnns(ann_ids)
            
            # Préparer les boîtes et classes
            boxes = []
            labels = []
            areas = []
            
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id in self.class_ids:
                    bbox = ann['bbox']
                    x1, y1, w, h = bbox
                    x2, y2 = x1 + w, y1 + h
                    
                    # Redimensionner les coordonnées selon la nouvelle taille d'image
                    orig_width = img_info['width']
                    orig_height = img_info['height']
                    
                    x1 = x1 * width / orig_width
                    y1 = y1 * height / orig_height
                    x2 = x2 * width / orig_width
                    y2 = y2 * height / orig_height
                    
                    # S'assurer que les coordonnées sont valides
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.cat_mapping[cat_id])
                        areas.append((x2 - x1) * (y2 - y1))
            
            # Convertir en tenseurs
            if boxes:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                areas = torch.as_tensor(areas, dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0), dtype=torch.int64)
                areas = torch.zeros((0), dtype=torch.float32)
            
            # Créer le dictionnaire cible
            target = {
                'boxes': boxes,
                'labels': labels,
                'area': areas,
                'image_id': torch.tensor([img_id]),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
            }
            
            # Appliquer les transformations
            if self.transform:
                img = self.transform(img)
            
            return img, target
            
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {img_id}: {e}")
            # Retourner une image et cible vides en cas d'erreur
            img = np.zeros((320, 320, 3), dtype=np.uint8)
            if self.transform:
                img = self.transform(img)
            
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0), dtype=torch.int64),
                'area': torch.zeros((0), dtype=torch.float32),
                'image_id': torch.tensor([img_id]),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
            
            return img, target

def collate_fn(batch):
    """Fonction de collation pour les batches"""
    return tuple(zip(*batch))

def get_faster_rcnn_model(num_classes):
    """Utilise Faster R-CNN (plus fiable que SSD MobileNet)"""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    # Charger le modèle pré-entraîné
    model = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Adapter la couche de classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    return model

def train_fast_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """Entraînement optimisé pour la vitesse"""
    model.to(device)
    
    # Optimiseur SGD
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler adaptatif
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Précision mixte corrigée
    scaler = GradScaler('cuda')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Mode entraînement
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Filtrer les images et cibles vides
            valid_data = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
            
            if not valid_data:
                continue  # Passer ce batch si aucune donnée valide
            
            images, targets = zip(*valid_data)
            images = list(images)
            targets = list(targets)
            
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            try:
                # Forward pass avec précision mixte
                with autocast('cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                # Vérifier les NaN
                if torch.isnan(losses):
                    print("NaN détecté, passage au batch suivant")
                    continue
                
                # Backward pass
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += losses.item()
                
                # Affichage
                progress_bar.set_postfix({
                    'loss': f"{losses.item():.3f}",
                    'avg_loss': f"{running_loss/(batch_idx+1):.3f}"
                })
                
            except Exception as e:
                print(f"Erreur dans le batch {batch_idx}: {e}")
                continue
        
        # Temps de l'époque
        epoch_time = time.time() - start_time
        images_per_sec = len(train_loader.dataset) / epoch_time
        
        print(f"Époque {epoch+1} terminée en {epoch_time:.1f}s")
        print(f"Vitesse: {images_per_sec:.1f} images/sec")
        
        # Validation simplifiée
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                if i >= 5:  # Limiter à 5 batches pour la validation
                    break
                
                # Filtrer les données valides
                valid_data = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
                
                if not valid_data:
                    continue
                
                images, targets = zip(*valid_data)
                images = list(images)
                targets = list(targets)
                
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                try:
                    model.train()  # Nécessaire pour obtenir les pertes
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                    val_batches += 1
                    model.eval()
                except Exception as e:
                    print(f"Erreur en validation: {e}")
                    continue
        
        if val_batches > 0:
            val_loss /= val_batches
            print(f"Validation Loss: {val_loss:.4f}")
        
        # Sauvegarder le modèle
        os.makedirs(config['output_dir'], exist_ok=True)
        torch.save(model.state_dict(), f"{config['output_dir']}/fast_model_epoch_{epoch+1}.pth")
        
        # Mettre à jour le learning rate
        scheduler.step()
        
        # Libérer la mémoire
        torch.cuda.empty_cache()

def main():
    # Chemins
    COCO_DIR = config['coco_dir']
    TRAIN_ANN_PATH = os.path.join(COCO_DIR, 'annotations', 'instances_train2017.json')
    VAL_ANN_PATH = os.path.join(COCO_DIR, 'annotations', 'instances_val2017.json')
    TRAIN_IMG_DIR = os.path.join(COCO_DIR, 'images', 'train2017')
    VAL_IMG_DIR = os.path.join(COCO_DIR, 'images', 'val2017')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Charger COCO
    print("Chargement des annotations COCO...")
    train_coco = COCO(TRAIN_ANN_PATH)
    val_coco = COCO(VAL_ANN_PATH)
    
    # Obtenir les classes
    class_ids = []
    for cls in config['classes']:
        cat_ids = train_coco.getCatIds(catNms=[cls])
        if cat_ids:
            class_ids.append(cat_ids[0])
    
    # Obtenir les images
    def find_images_with_objects(coco, class_ids):
        img_ids = set()
        for class_id in class_ids:
            ids = coco.getImgIds(catIds=[class_id])
            img_ids.update(ids)
        return sorted(list(img_ids))
    
    train_img_ids = find_images_with_objects(train_coco, class_ids)
    val_img_ids = find_images_with_objects(val_coco, class_ids)
    
    val_img_ids, test_img_ids = train_test_split(val_img_ids, test_size=0.3, random_state=42)
    
    # Limiter le nombre d'images
    train_img_ids = train_img_ids[:config['max_train_images']]
    val_img_ids = val_img_ids[:config['max_val_images']]
    
    print(f"Images limitées - Train: {len(train_img_ids)}, Val: {len(val_img_ids)}")
    
    # Transformations simplifiées
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Créer les datasets
    train_dataset = LostObjectsDatasetForPretrained(
        train_coco, train_img_ids, TRAIN_IMG_DIR, class_ids, transform
    )
    
    val_dataset = LostObjectsDatasetForPretrained(
        val_coco, val_img_ids, VAL_IMG_DIR, class_ids, transform
    )
    
    # DataLoaders simplifiés (sans multiprocessing pour éviter les erreurs de mémoire)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Créer le modèle
    print("Chargement du modèle Faster R-CNN...")
    model = get_faster_rcnn_model(len(config['classes']))
    print("Modèle chargé avec succès!")
    
    # Entraîner
    train_fast_model(model, train_loader, val_loader, config['num_epochs'], device)

if __name__ == "__main__":
    main()