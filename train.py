import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Forcer l'utilisation du GPU NVIDIA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Optimisations CUDA
torch.backends.cudnn.benchmark = True

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
        
        # Charger les informations de l'image
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Charger l'image
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            # Image de secours si le chargement échoue
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
        areas = []
        
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in self.class_ids:
                bbox = ann['bbox']
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
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

def collate_fn(batch):
    return tuple(zip(*batch))

def get_pretrained_model(num_classes):
    """
    Charge un modèle Faster R-CNN pré-entraîné et l'adapte à nos classes
    """
    # Charger un modèle pré-entraîné
    model = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Remplacer la couche de classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 pour le fond
    
    return model

def train_model(model, train_loader, val_loader, num_epochs=15, device='cuda', output_dir='output_pretrained'):
    """
    Entraîne le modèle pré-entraîné
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.to(device)
    
    # Paramètres d'optimisation
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        # Mode entraînement
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Réinitialiser les gradients
            optimizer.zero_grad()
            
            # Forward pass - les modèles torchvision retournent automatiquement les pertes
            loss_dict = model(images, targets)
            
            # Calculer la perte totale
            losses = sum(loss for loss in loss_dict.values())
            
            # Vérifier les NaN
            if torch.isnan(losses):
                print("NaN détecté, passage au batch suivant")
                continue
            
            # Backward pass
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
            
            # Mettre à jour la barre de progression
            progress_bar.set_postfix({
                'loss': losses.item(),
                'cls_loss': loss_dict.get('loss_classifier', 0).item(),
                'bbox_loss': loss_dict.get('loss_box_reg', 0).item()
            })
        
        # Calculer la perte moyenne de l'époque
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        print(f"Train Loss: {epoch_loss:.4f}")
        
        # Mode évaluation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass en mode évaluation
                model.train()  # Nécessaire pour obtenir les pertes
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                model.eval()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Mettre à jour le learning rate
        lr_scheduler.step()
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_pretrained_model.pth'))
            print("Meilleur modèle sauvegardé!")
        
        # Sauvegarder après chaque époque
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': val_loss,
        }, os.path.join(output_dir, f'pretrained_epoch_{epoch+1}.pth'))
        
        # Graphique des pertes
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epoch+2), train_losses, 'b-', label='Training Loss')
        plt.plot(range(1, epoch+2), val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses (Pretrained Model)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        plt.close()

def main():
    from config import config
    
    # Paramètres
    num_classes = len(config['classes'])
    batch_size = 4  # Réduire le batch size pour les modèles pré-entraînés
    num_epochs = 15  # Moins d'époques nécessaires
    
    # Chemins des données
    COCO_DIR = config['coco_dir']
    TRAIN_ANN_PATH = os.path.join(COCO_DIR, 'annotations', 'instances_train2017.json')
    VAL_ANN_PATH = os.path.join(COCO_DIR, 'annotations', 'instances_val2017.json')
    TRAIN_IMG_DIR = os.path.join(COCO_DIR, 'images', 'train2017')
    VAL_IMG_DIR = os.path.join(COCO_DIR, 'images', 'val2017')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Charger les données COCO
    train_coco = COCO(TRAIN_ANN_PATH)
    val_coco = COCO(VAL_ANN_PATH)
    
    # Obtenir les IDs de classes
    class_ids = []
    for cls in config['classes']:
        cat_ids = train_coco.getCatIds(catNms=[cls])
        if cat_ids:
            class_ids.append(cat_ids[0])
    
    # Obtenir les images contenant nos objets
    def find_images_with_objects(coco, class_ids):
        img_ids = set()
        for class_id in class_ids:
            ids = coco.getImgIds(catIds=[class_id])
            img_ids.update(ids)
        return sorted(list(img_ids))
    
    train_img_ids = find_images_with_objects(train_coco, class_ids)
    val_img_ids = find_images_with_objects(val_coco, class_ids)
    
    # Diviser validation en val et test
    val_img_ids, test_img_ids = train_test_split(val_img_ids, test_size=0.3, random_state=42)
    
    print(f"Images - Train: {len(train_img_ids)}, Val: {len(val_img_ids)}")
    
    # Transformations (plus simples pour les modèles pré-entraînés)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Créer les datasets
    train_dataset = LostObjectsDatasetForPretrained(
        train_coco, train_img_ids, TRAIN_IMG_DIR, class_ids, train_transform
    )
    
    val_dataset = LostObjectsDatasetForPretrained(
        val_coco, val_img_ids, VAL_IMG_DIR, class_ids, val_transform
    )
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # Créer le modèle pré-entraîné
    model = get_pretrained_model(num_classes)
    
    print("Modèle pré-entraîné chargé avec succès!")
    print(f"Entraînement sur {num_classes} classes: {config['classes']}")
    
    # Entraîner le modèle
    train_model(model, train_loader, val_loader, num_epochs, device)

if __name__ == "__main__":
    main()