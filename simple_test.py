import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.model import ObjectDetectionModel
from config import config

# Forcer l'utilisation du GPU NVIDIA uniquement pour le forward pass
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(model_path, num_classes):
    """Charge le modèle en toute sécurité"""
    model = ObjectDetectionModel(num_classes)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modèle chargé depuis checkpoint (époque {checkpoint['epoch']})")
    else:
        model.load_state_dict(checkpoint)
        print("Modèle chargé")
    
    return model

def detect_single_image(image_path, model, device, class_names, threshold=0.3):
    """
    Fonction simplifiée pour détecter les objets dans une image
    """
    # Charger et prétraiter l'image
    image = Image.open(image_path).convert('RGB')
    
    # Préserver les dimensions originales
    width, height = image.size
    
    # Prétraitement
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inférence
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Forward pass uniquement (pas de post-traitement avec CUDA)
        cls_preds, reg_preds = model(input_tensor)
        
        # Passer immédiatement au CPU pour éviter les erreurs CUDA
        cls_preds = cls_preds.cpu()
        reg_preds = reg_preds.cpu()
    
    # Extraire les scores de confiance et les indices des classes
    cls_probs = torch.softmax(cls_preds, dim=2)
    
    # Plus besoin de max() qui peut causer des problèmes
    # On garde les probabilités pour toutes les classes
    
    # Créer une image de sortie
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Déterminer les emplacements des caractéristiques
    # (Version très simplifiée - pas de décodage des boîtes)
    # Nous allons juste visualiser les heatmaps de confiance
    
    plt.figure(figsize=(15, 10))
    
    # Affichage de l'image originale
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image d'entrée")
    plt.axis('off')
    
    # Affichage des heatmaps de confiance pour quelques classes
    plt.subplot(1, 2, 2)
    
    # Sélectionner quelques classes (exclure la classe de fond)
    confidences = []
    for i in range(1, min(len(class_names) + 1, model.num_classes)):
        max_conf = cls_probs[0, :, i].max().item()
        confidences.append((class_names[i-1], max_conf))
    
    # Trier par confiance
    confidences.sort(key=lambda x: x[1], reverse=True)
    
    # Afficher le résumé
    plt.axis('off')
    plt.text(0.1, 0.9, "Probabilités maximales détectées:", fontsize=14)
    
    for i, (cls_name, conf) in enumerate(confidences):
        color = 'green' if conf > threshold else 'gray'
        plt.text(0.1, 0.8 - i*0.05, f"{cls_name}: {conf:.4f}", 
                 fontsize=12, color=color)
    
    plt.title("Résumé des détections")
    
    # Sauvegarder le résultat
    output_dir = 'detection_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"detect_{os.path.basename(image_path)}"))
    print(f"Résultat sauvegardé pour {image_path}")
    
    return confidences

def main():
    # Paramètres
    num_classes = len(config['classes'])
    class_names = config['classes']
    
    # Utiliser une époque avec de bonnes performances (pas une avec des NaN)
    model_path = os.path.join(config['output_dir'], 'model_epoch_4.pth')
    
    if not os.path.exists(model_path):
        print(f"Modèle non trouvé: {model_path}")
        return
    
    # Utiliser CPU ou GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # Charger le modèle
    model = load_model(model_path, num_classes)
    
    # Dossier de test
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)
    
    # Liste des images de test
    import glob
    test_images = glob.glob(os.path.join(test_dir, '*.jpg')) + \
                 glob.glob(os.path.join(test_dir, '*.png')) + \
                 glob.glob(os.path.join(test_dir, '*.jpeg'))
    
    if not test_images:
        print(f"Aucune image trouvée dans {test_dir}")
        return
    
    print(f"Traitement de {len(test_images)} images...")
    
    # Traiter chaque image
    for img_path in test_images:
        print(f"Analyse de: {os.path.basename(img_path)}")
        try:
            confidences = detect_single_image(img_path, model, device, class_names)
            print("Confidences détectées:")
            for cls, conf in confidences:
                print(f"  {cls}: {conf:.4f}")
        except Exception as e:
            print(f"Erreur lors du traitement de {img_path}: {e}")

if __name__ == "__main__":
    main()