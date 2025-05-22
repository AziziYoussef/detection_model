import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models.detection as detection_models
from torch.amp import GradScaler, autocast
import time
import cv2
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import de la nouvelle configuration
from config_extended import config

# Optimisations CUDA
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LostObjectsDatasetExtended(torch.utils.data.Dataset):
    """Dataset pour 30 classes d'objets perdus"""
    
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
                img = np.zeros((320, 320, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionner l'image
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
                    
                    # Redimensionner les coordonnées
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
    """Créer le modèle Faster R-CNN pour 30 classes"""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    # Charger le modèle pré-entraîné
    model = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Adapter la couche de classification pour 30 classes + fond
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    return model

def train_extended_model(model, train_loader, val_loader, num_epochs, device):
    """Entraînement du modèle avec 30 classes"""
    model.to(device)
    
    # Optimiseur avec learning rate adapté
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler pour réduire le learning rate
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 15], gamma=0.1)
    
    # Précision mixte
    scaler = GradScaler('cuda')
    
    # Historique des pertes
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("="*50)
        
        # Mode entraînement
        model.train()
        running_loss = 0.0
        valid_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Filtrer les images et cibles vides
            valid_data = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
            
            if not valid_data:
                continue
            
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
                valid_batches += 1
                
                # Affichage détaillé
                progress_bar.set_postfix({
                    'loss': f"{losses.item():.3f}",
                    'cls': f"{loss_dict.get('loss_classifier', 0):.3f}",
                    'box': f"{loss_dict.get('loss_box_reg', 0):.3f}",
                    'obj': f"{loss_dict.get('loss_objectness', 0):.3f}",
                    'rpn': f"{loss_dict.get('loss_rpn_box_reg', 0):.3f}"
                })
                
            except Exception as e:
                print(f"Erreur dans le batch {batch_idx}: {e}")
                continue
        
        # Calculer la perte moyenne de l'époque
        if valid_batches > 0:
            epoch_loss = running_loss / valid_batches
            train_losses.append(epoch_loss)
        else:
            epoch_loss = float('inf')
            train_losses.append(epoch_loss)
        
        # Temps de l'époque
        epoch_time = time.time() - start_time
        print(f"Train Loss: {epoch_loss:.4f}")
        print(f"Temps d'époque: {epoch_time:.1f}s")
        
        # Validation simplifiée
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(val_loader):
                if i >= 10:  # Limiter à 10 batches pour la validation
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
                    continue
        
        if val_batches > 0:
            val_loss /= val_batches
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4f}")
        
        # Sauvegarder le modèle
        os.makedirs(config['output_dir'], exist_ok=True)
        torch.save(model.state_dict(), f"{config['output_dir']}/extended_model_epoch_{epoch+1}.pth")
        
        # Sauvegarder le meilleur modèle
        if epoch == 0 or (val_batches > 0 and val_loss < min(val_losses[:-1] + [float('inf')])):
            torch.save(model.state_dict(), f"{config['output_dir']}/best_extended_model.pth")
            print("🏆 Meilleur modèle sauvegardé!")
        
        # Mettre à jour le learning rate
        scheduler.step()
        
        # Graphique des pertes
        if len(train_losses) > 1:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
            if len(val_losses) > 0:
                plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress - 30 Classes Extended Model')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{config['output_dir']}/training_progress.png")
            plt.close()
        
        # Libérer la mémoire
        torch.cuda.empty_cache()

def main():
    print("="*60)
    print("ENTRAÎNEMENT MODÈLE ÉTENDU - 30 CLASSES D'OBJETS PERDUS")
    print("="*60)
    
    # Chemins
    COCO_DIR = config['coco_dir']
    TRAIN_ANN_PATH = os.path.join(COCO_DIR, 'annotations', 'instances_train2017.json')
    VAL_ANN_PATH = os.path.join(COCO_DIR, 'annotations', 'instances_val2017.json')
    TRAIN_IMG_DIR = os.path.join(COCO_DIR, 'images', 'train2017')
    VAL_IMG_DIR = os.path.join(COCO_DIR, 'images', 'val2017')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilisé: {device}")
    print(f"Nombre de classes: {config['num_classes']}")
    
    # Charger COCO
    print("Chargement des annotations COCO...")
    train_coco = COCO(TRAIN_ANN_PATH)
    val_coco = COCO(VAL_ANN_PATH)
    
    # Obtenir les IDs de classes
    class_ids = []
    missing_classes = []
    
    for cls in config['classes']:
        cat_ids = train_coco.getCatIds(catNms=[cls])
        if cat_ids:
            class_ids.append(cat_ids[0])
            print(f"✓ {cls}")
        else:
            missing_classes.append(cls)
            print(f"✗ {cls} (non trouvé dans COCO)")
    
    if missing_classes:
        print(f"\nClasses manquantes: {missing_classes}")
        print("Le modèle sera entraîné avec les classes disponibles seulement.")
    
    print(f"\nClasses finales: {len(class_ids)} classes disponibles")
    
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
    
    print(f"Images - Train: {len(train_img_ids)}, Val: {len(val_img_ids)}")
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Créer les datasets
    train_dataset = LostObjectsDatasetExtended(
        train_coco, train_img_ids, TRAIN_IMG_DIR, class_ids, transform
    )
    
    val_dataset = LostObjectsDatasetExtended(
        val_coco, val_img_ids, VAL_IMG_DIR, class_ids, transform
    )
    
    # DataLoaders
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
    print("Chargement du modèle Faster R-CNN étendu...")
    model = get_faster_rcnn_model(len(class_ids))
    print("Modèle chargé avec succès!")
    
    # Entraîner
    train_extended_model(model, train_loader, val_loader, config['num_epochs'], device)
    
    print("\n🎉 ENTRAÎNEMENT TERMINÉ!")
    print(f"Modèles sauvegardés dans: {config['output_dir']}/")

if __name__ == "__main__":
    main()