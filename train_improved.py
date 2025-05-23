# train_stable.py - Version ultra-stabilisée pour éviter les NaN
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models.detection as detection_models
import time
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import de la configuration stabilisée
from config_improved import config

# Optimisations CUDA
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class StableDataset(torch.utils.data.Dataset):
    """Dataset ultra-stable avec vérifications"""
    
    def __init__(self, coco, img_ids, img_dir, class_ids, config, transform=None):
        self.coco = coco
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.class_ids = class_ids
        self.config = config
        self.transform = transform
        
        # Mapping de catégories COCO vers nos indices
        self.cat_mapping = {cat_id: i + 1 for i, cat_id in enumerate(class_ids)}
        
        print(f"Dataset créé avec {len(img_ids)} images et {len(class_ids)} classes")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        try:
            # Charger l'image avec PIL (plus stable)
            img_info = self.coco.loadImgs([img_id])[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                # Image de secours plus petite
                img = Image.new('RGB', (320, 320), color=(128, 128, 128))
            
            # Redimensionner avec PIL
            img = img.resize(self.config['image_size'], Image.Resampling.BILINEAR)
            img = np.array(img)
            
            height, width = img.shape[:2]
            
            # Récupérer annotations avec vérifications
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.class_ids)
            anns = self.coco.loadAnns(ann_ids)
            
            boxes = []
            labels = []
            areas = []
            
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id in self.class_ids and 'bbox' in ann:
                    bbox = ann['bbox']
                    x1, y1, w, h = bbox
                    
                    # VÉRIFICATIONS STRICTES pour éviter les NaN
                    if w <= 0 or h <= 0:
                        continue
                    
                    x2, y2 = x1 + w, y1 + h
                    
                    # Adapter aux nouvelles dimensions
                    orig_width = max(1, img_info['width'])  # Éviter division par 0
                    orig_height = max(1, img_info['height'])
                    
                    scale_x = width / orig_width
                    scale_y = height / orig_height
                    
                    x1 = x1 * scale_x
                    y1 = y1 * scale_y
                    x2 = x2 * scale_x
                    y2 = y2 * scale_y
                    
                    # VALIDATION STRICTE des coordonnées
                    x1 = max(0, min(x1, width - 2))
                    y1 = max(0, min(y1, height - 2))
                    x2 = max(x1 + 2, min(x2, width))
                    y2 = max(y1 + 2, min(y2, height))
                    
                    # Vérifier la taille finale
                    final_w = x2 - x1
                    final_h = y2 - y1
                    
                    if final_w >= 2 and final_h >= 2:  # Taille minimum absolue
                        # Filtrage spécial pour les personnes
                        if cat_id == self.coco.getCatIds(catNms=['person'])[0]:
                            min_size = self.config['person_detection_params']['min_person_size']
                            if final_w < min_size or final_h < min_size:
                                continue
                        
                        # Vérifier que les valeurs ne sont pas NaN ou inf
                        if all(np.isfinite([x1, y1, x2, y2])):
                            boxes.append([float(x1), float(y1), float(x2), float(y2)])
                            labels.append(int(self.cat_mapping[cat_id]))
                            areas.append(float(final_w * final_h))
            
            # SÉCURITÉ: Toujours avoir au moins une boîte valide ou retourner vide
            if len(boxes) == 0:
                # Créer une boîte factice très petite
                boxes = [[1.0, 1.0, 3.0, 3.0]]
                labels = [1]  # Classe par défaut
                areas = [4.0]
            
            # Convertir en tenseurs avec vérifications
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            
            # VÉRIFICATION FINALE des tenseurs
            if torch.any(torch.isnan(boxes)) or torch.any(torch.isnan(areas)):
                print(f"⚠️ NaN détecté dans les tenseurs pour image {img_id}")
                # Remplacer par des valeurs sûres
                boxes = torch.tensor([[1.0, 1.0, 3.0, 3.0]], dtype=torch.float32)
                labels = torch.tensor([1], dtype=torch.int64)
                areas = torch.tensor([4.0], dtype=torch.float32)
            
            # Target dict
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
            print(f"❌ Erreur critique image {img_id}: {e}")
            # Retour de secours complet
            img = np.zeros((*self.config['image_size'], 3), dtype=np.uint8)
            if self.transform:
                img = self.transform(img)
            
            target = {
                'boxes': torch.tensor([[1.0, 1.0, 3.0, 3.0]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'area': torch.tensor([4.0], dtype=torch.float32),
                'image_id': torch.tensor([img_id]),
                'iscrowd': torch.zeros((1,), dtype=torch.int64)
            }
            
            return img, target

def get_stable_model(num_classes):
    """Créer un modèle avec initialisation stable"""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    print("🤖 Chargement du modèle Faster R-CNN...")
    model = detection_models.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Adapter la couche de classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    # INITIALISATION STABLE des nouvelles couches
    for module in model.roi_heads.box_predictor.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.01)
            torch.nn.init.constant_(module.bias, 0)
    
    return model

def check_for_nan(tensor, name="tensor"):
    """Vérifier les NaN dans un tenseur"""
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN détecté dans {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf détecté dans {name}")
        return True
    return False

def safe_backward_pass(loss, model, optimizer, scaler, config):
    """Backward pass sécurisé avec vérifications"""
    
    # Vérifier la loss avant backward
    if check_for_nan(loss, "loss"):
        return False
    
    if loss.item() > 100:  # Loss trop élevée
        print(f"⚠️ Loss très élevée: {loss.item():.3f}")
        return False
    
    try:
        # Backward pass
        if config.get('use_mixed_precision', False):
            scaler.scale(loss).backward()
            
            # Vérifier les gradients
            if config.get('check_gradients', False):
                for name, param in model.named_parameters():
                    if param.grad is not None and check_for_nan(param.grad, f"grad_{name}"):
                        return False
            
            # Gradient clipping
            if config.get('gradient_clipping', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Mode normal sans mixed precision
            loss.backward()
            
            # Vérifier les gradients
            if config.get('check_gradients', False):
                for name, param in model.named_parameters():
                    if param.grad is not None and check_for_nan(param.grad, f"grad_{name}"):
                        return False
            
            # Gradient clipping
            if config.get('gradient_clipping', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping'])
            
            optimizer.step()
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans backward pass: {e}")
        return False

def train_stable_model(model, train_loader, val_loader, config, device):
    """Entraînement ultra-stabilisé"""
    model.to(device)
    
    # Optimiseur SGD (plus stable)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler avec warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return float(epoch) / float(max(1, config['warmup_epochs']))
        return 0.1 ** (epoch // 10)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Scaler pour mixed precision (si activé)
    scaler = torch.cuda.amp.GradScaler() if config.get('use_mixed_precision', False) else None
    
    # Historique
    train_losses = []
    best_loss = float('inf')
    nan_count = 0
    
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT STABILISÉ")
    print("="*60)
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Mixed Precision: {config.get('use_mixed_precision', False)}")
    print(f"Gradient Clipping: {config.get('gradient_clipping', 0)}")
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        print(f"\n📅 Epoch {epoch+1}/{config['num_epochs']}")
        print("="*50)
        
        # === MODE ENTRAÎNEMENT ===
        model.train()
        running_loss = 0.0
        valid_batches = 0
        skipped_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Filtrer les données valides avec vérifications strictes
            valid_data = []
            for img, tgt in zip(images, targets):
                if (len(tgt['boxes']) > 0 and 
                    not torch.isnan(tgt['boxes']).any() and 
                    not torch.isnan(tgt['area']).any()):
                    valid_data.append((img, tgt))
            
            if not valid_data:
                skipped_batches += 1
                continue
            
            images, targets = zip(*valid_data)
            images = list(images)
            targets = list(targets)
            
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                if config.get('use_mixed_precision', False):
                    with torch.cuda.amp.autocast():
                        loss_dict = model(images, targets)
                        total_loss = sum(loss for loss in loss_dict.values())
                else:
                    loss_dict = model(images, targets)
                    total_loss = sum(loss for loss in loss_dict.values())
                
                # Vérifier la loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"⚠️ NaN/Inf détecté dans la loss au batch {batch_idx}")
                    nan_count += 1
                    if nan_count > 20:  # Trop de NaN
                        print("❌ Trop de NaN détectés, arrêt de l'entraînement")
                        return model
                    continue
                
                # Backward pass sécurisé
                success = safe_backward_pass(total_loss, model, optimizer, scaler, config)
                
                if success:
                    running_loss += total_loss.item()
                    valid_batches += 1
                    
                    # Affichage
                    progress_bar.set_postfix({
                        'loss': f"{total_loss.item():.3f}",
                        'cls': f"{loss_dict.get('loss_classifier', 0):.3f}",
                        'box': f"{loss_dict.get('loss_box_reg', 0):.3f}",
                        'skipped': skipped_batches,
                        'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                    })
                else:
                    skipped_batches += 1
                    
            except Exception as e:
                print(f"❌ Erreur batch {batch_idx}: {e}")
                skipped_batches += 1
                continue
        
        # Calculer la perte moyenne
        if valid_batches > 0:
            epoch_loss = running_loss / valid_batches
            train_losses.append(epoch_loss)
        else:
            print("❌ Aucun batch valide dans cette époque!")
            continue
        
        # Temps d'époque
        epoch_time = time.time() - start_time
        print(f"🏃 Epoch Loss: {epoch_loss:.4f} | Temps: {epoch_time:.1f}s")
        print(f"📊 Batches valides: {valid_batches}, Ignorés: {skipped_batches}")
        
        # Sauvegarder si meilleur
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"{config['output_dir']}/best_stable_model.pth")
            print("🏆 Meilleur modèle sauvegardé!")
        
        # Sauvegarder régulièrement
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{config['output_dir']}/stable_model_epoch_{epoch+1}.pth")
        
        # Mettre à jour le learning rate
        scheduler.step()
        
        # Graphique des pertes
        if len(train_losses) > 1:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Progression de l\'entraînement stabilisé')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Échelle log pour mieux voir
            plt.savefig(f"{config['output_dir']}/training_progress_stable.png")
            plt.close()
        
        # Nettoyage mémoire
        torch.cuda.empty_cache()
    
    print("\n🎉 ENTRAÎNEMENT TERMINÉ!")
    return model

def collate_fn(batch):
    """Fonction de collation sécurisée"""
    return tuple(zip(*batch))

def main():
    print("="*80)
    print("🛡️ ENTRAÎNEMENT ULTRA-STABILISÉ - ANTI-NaN")
    print("="*80)
    
    # Créer le dossier de sortie
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Chemins
    COCO_DIR = config['coco_dir']
    TRAIN_ANN_PATH = os.path.join(COCO_DIR, 'annotations', 'instances_train2017.json')
    VAL_ANN_PATH = os.path.join(COCO_DIR, 'annotations', 'instances_val2017.json')
    TRAIN_IMG_DIR = os.path.join(COCO_DIR, 'images', 'train2017')
    VAL_IMG_DIR = os.path.join(COCO_DIR, 'images', 'val2017')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    print(f"📊 Classes configurées: {config['num_classes']}")
    
    # Charger COCO
    print("📋 Chargement des annotations COCO...")
    train_coco = COCO(TRAIN_ANN_PATH)
    val_coco = COCO(VAL_ANN_PATH)
    
    # Obtenir les IDs de classes
    class_ids = []
    found_classes = []
    
    for cls in config['classes']:
        cat_ids = train_coco.getCatIds(catNms=[cls])
        if cat_ids:
            class_ids.append(cat_ids[0])
            found_classes.append(cls)
            print(f"✅ {cls}")
        else:
            print(f"❌ {cls} (non trouvé)")
    
    print(f"\n📈 RÉSUMÉ:")
    print(f"  ✅ Classes trouvées: {len(found_classes)}")
    
    # Vérifier la détection de personnes
    if 'person' in found_classes:
        print("👥 Détection de personnes: ✅ ACTIVÉE")
    else:
        print("⚠️ 'person' non trouvé dans COCO!")
    
    # Obtenir les images
    def find_images_with_objects(coco, class_ids):
        img_ids = set()
        for class_id in class_ids:
            ids = coco.getImgIds(catIds=[class_id])[:2000]  # Limiter pour stabilité
            img_ids.update(ids)
        return sorted(list(img_ids))
    
    train_img_ids = find_images_with_objects(train_coco, class_ids)
    val_img_ids = find_images_with_objects(val_coco, class_ids)
    
    # Split validation
    val_img_ids, test_img_ids = train_test_split(val_img_ids, test_size=0.3, random_state=42)
    
    # Limiter les images
    train_img_ids = train_img_ids[:config['max_train_images']]
    val_img_ids = val_img_ids[:config['max_val_images']]
    
    print(f"📷 Images - Train: {len(train_img_ids)}, Val: {len(val_img_ids)}")
    
    # Transformations minimales (plus stable)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets stabilisés
    train_dataset = StableDataset(
        train_coco, train_img_ids, TRAIN_IMG_DIR, class_ids, config, train_transform
    )
    
    val_dataset = StableDataset(
        val_coco, val_img_ids, VAL_IMG_DIR, class_ids, config, val_transform
    )
    
    # DataLoaders avec paramètres conservateurs
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True  # Éviter les batches incomplets
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    # Créer le modèle stabilisé
    print("🤖 Chargement du modèle stabilisé...")
    model = get_stable_model(len(class_ids))
    print("✅ Modèle stabilisé chargé!")
    
    # Entraîner
    trained_model = train_stable_model(model, train_loader, val_loader, config, device)
    
    print(f"\n🎊 ENTRAÎNEMENT STABILISÉ TERMINÉ!")
    print(f"📁 Modèles sauvegardés dans: {config['output_dir']}/")
    print(f"🏆 Meilleur modèle: best_stable_model.pth")

if __name__ == "__main__":
    main()