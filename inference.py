import os
# Forcer l'utilisation du GPU NVIDIA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.model import ObjectDetectionModel
from models.anchors import AnchorGenerator
from utils.box_utils import decode_boxes, nms
from config import config

def load_model(model_path, num_classes, device):
    """
    Charge un modèle entraîné
    
    Args:
        model_path (str): Chemin vers le fichier de modèle
        num_classes (int): Nombre de classes (sans le fond)
        device (torch.device): Appareil pour les tenseurs
        
    Returns:
        nn.Module: Modèle chargé
    """
    model = ObjectDetectionModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, image_size=(512, 512)):
    """
    Prétraite une image pour l'inférence
    
    Args:
        image_path (str): Chemin vers l'image
        image_size (tuple): Taille de l'image (height, width)
        
    Returns:
        tuple: (image_tensor, original_image, scale_factors)
    """
    # Charger l'image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Calculer les facteurs d'échelle
    height, width = original_image.shape[:2]
    scale_h, scale_w = image_size[0] / height, image_size[1] / width
    
    # Redimensionner l'image
    image = cv2.resize(original_image, (image_size[1], image_size[0]))
    
    # Normaliser l'image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    
    return image_tensor, original_image, (scale_h, scale_w)

def detect_objects(model, image_tensor, anchors, class_names, conf_thresh=0.5, nms_thresh=0.5, device='cuda'):
    """
    Détecte les objets dans une image
    
    Args:
        model (nn.Module): Modèle de détection
        image_tensor (torch.Tensor): Image prétraitée
        anchors (torch.Tensor): Anchors
        class_names (list): Noms des classes
        conf_thresh (float): Seuil de confiance
        nms_thresh (float): Seuil NMS
        device (torch.device): Appareil pour les tenseurs
        
    Returns:
        tuple: (boxes, labels, scores)
    """
    # Ajouter une dimension de batch
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        cls_preds, reg_preds = model(image_tensor)
    
    # Obtenir les scores et les classes
    scores, labels = torch.max(F.softmax(cls_preds, dim=-1), dim=-1)
    
    # Décoder les boîtes
    boxes = decode_boxes(reg_preds, anchors.to(device))[0]  # [num_anchors, 4]
    
    # Filtrer les prédictions par score
    mask = scores[0] > conf_thresh
    boxes = boxes[mask]
    labels = labels[0][mask]
    scores = scores[0][mask]
    
    # Appliquer NMS pour chaque classe
    keep_boxes = []
    keep_labels = []
    keep_scores = []
    
    for c in range(1, model.num_classes):  # Ignorer la classe de fond (0)
        class_mask = labels == c
        if not class_mask.any():
            continue
            
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        # Appliquer NMS
        keep_indices = nms(class_boxes, class_scores, nms_thresh)
        
        if len(keep_indices) > 0:
            keep_boxes.append(class_boxes[keep_indices])
            keep_labels.append(labels[class_mask][keep_indices])
            keep_scores.append(class_scores[keep_indices])
    
    if not keep_boxes:
        return torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long), torch.zeros(0)
    
    # Concaténer les résultats
    boxes = torch.cat(keep_boxes, dim=0)
    labels = torch.cat(keep_labels, dim=0)
    scores = torch.cat(keep_scores, dim=0)
    
    return boxes.cpu(), labels.cpu(), scores.cpu()

def visualize_detections(image, boxes, labels, scores, class_names, scale_factors=None, output_path=None):
    """
    Visualise les détections sur l'image
    
    Args:
        image (numpy.ndarray): Image originale
        boxes (torch.Tensor): Boîtes détectées [N, 4]
        labels (torch.Tensor): Étiquettes [N]
        scores (torch.Tensor): Scores [N]
        class_names (list): Noms des classes
        scale_factors (tuple): Facteurs d'échelle (scale_h, scale_w)
        output_path (str): Chemin pour sauvegarder l'image (optionnel)
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    ax = plt.gca()
    
    # Générer des couleurs pour chaque classe
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names) + 1))
    
    # Convertir les boîtes de coordonnées normalisées à coordonnées pixels
    if scale_factors:
        scale_h, scale_w = scale_factors
        h, w = image.shape[:2]
        
        # Dénormaliser les boîtes
        boxes_pixel = boxes.clone()
        boxes_pixel[:, [0, 2]] *= w / scale_w
        boxes_pixel[:, [1, 3]] *= h / scale_h
    else:
        h, w = image.shape[:2]
        boxes_pixel = boxes.clone()
        boxes_pixel[:, [0, 2]] *= w
        boxes_pixel[:, [1, 3]] *= h
    
    # Dessiner chaque détection
    for box, label, score in zip(boxes_pixel, labels, scores):
        x1, y1, x2, y2 = box.numpy()
        
        # Obtenir la classe et la couleur
        class_idx = label.item() - 1  # -1 car l'indice 0 est la classe de fond
        if class_idx < 0 or class_idx >= len(class_names):
            continue
            
        class_name = class_names[class_idx]
        color = colors[class_idx]
        
        # Créer un rectangle
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Ajouter le texte
        plt.text(
            x1, y1-5, f"{class_name}: {score:.2f}",
            color='white', fontsize=10,
            bbox=dict(facecolor=color, alpha=0.8, pad=0)
        )
    
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    plt.show()

def main():
    # Charger la configuration
    model_path = os.path.join(config['output_dir'], 'model_epoch_5.pth')  # Utilise le meilleur modèle
    image_path = 'test_images/test_image.jpg'  # Remplacez par le chemin de votre image de test
    num_classes = len(config['classes'])
    image_size = config['image_size']
    conf_thresh = 0.5
    nms_thresh = 0.5
    # Pour tester plusieurs images, décommentez ce code
    """
    import glob
    test_images = glob.glob('test_images/*.jpg')
    for image_path in test_images:
        print(f"\nTesting: {image_path}")
        # Prétraiter l'image
        image_tensor, original_image, scale_factors = preprocess_image(image_path, image_size)
        # Détecter les objets
        boxes, labels, scores = detect_objects(model, image_tensor, anchors, class_names, conf_thresh, nms_thresh, device)
        # Visualiser les détections
        output_path = os.path.join(config['output_dir'], f'detection_{os.path.basename(image_path)}')
        visualize_detections(original_image, boxes, labels, scores, class_names, scale_factors, output_path)
    """
    # Définir le device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        print(f"Le modèle n'existe pas à {model_path}, veuillez d'abord entraîner le modèle.")
        return
    
    # Vérifier si l'image de test existe
    if not os.path.exists(image_path):
        print(f"L'image de test n'existe pas à {image_path}, veuillez spécifier un chemin valide.")
        return
    
    # Charger le modèle
    model = load_model(model_path, num_classes, device)
    
    # Définir le générateur d'anchors
    anchor_generator = AnchorGenerator(
        sizes=config['anchor_sizes'],
        aspect_ratios=config['anchor_aspect_ratios'],
        strides=config['anchor_strides']
    )
    
    # Générer les anchors
    anchors = anchor_generator.generate_anchors(image_size)
    
    # Prétraiter l'image
    image_tensor, original_image, scale_factors = preprocess_image(image_path, image_size)
    
    # Détecter les objets
    boxes, labels, scores = detect_objects(
        model, image_tensor, anchors, config['classes'], conf_thresh, nms_thresh, device
    )
    
    print(f"Détections: {len(boxes)} objets trouvés")
    
    # Visualiser les détections
    if len(boxes) > 0:
        output_path = os.path.join(config['output_dir'], 'detection_result.png')
        visualize_detections(
            original_image, boxes, labels, scores, config['classes'], 
            scale_factors, output_path
        )
        print(f"Résultat de détection sauvegardé à {output_path}")
    else:
        print("Aucun objet détecté!")

if __name__ == "__main__":
    main()