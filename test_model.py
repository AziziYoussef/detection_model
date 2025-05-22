import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from config import config

def load_pretrained_model(model_path, num_classes):
    """Charge le modèle pré-entraîné"""
    model = detection_models.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Adapter la couche de classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    # Charger les poids
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def test_image(model, image_path, class_names, device, conf_thresh=0.5):
    """Teste une image avec le modèle pré-entraîné"""
    # Charger et prétraiter l'image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inférence
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Extraire les résultats
    boxes = predictions[0]['boxes'].cpu()
    labels = predictions[0]['labels'].cpu()
    scores = predictions[0]['scores'].cpu()
    
    # Filtrer par score
    mask = scores > conf_thresh
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    return boxes, labels, scores, np.array(image)

def visualize_detections(image, boxes, labels, scores, class_names):
    """Visualise les détections"""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.numpy()
        
        # Obtenir le nom de la classe (label-1 car 0 est le fond)
        class_idx = label.item() - 1
        if 0 <= class_idx < len(class_names):
            class_name = class_names[class_idx]
            color = colors[class_idx]
            
            # Dessiner la boîte
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Ajouter le texte
            plt.text(x1, y1-5, f"{class_name}: {score:.2f}",
                    color='white', fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Paramètres
    model_path = 'output_fast/fast_model_epoch_10.pth'
    test_dir = 'test_images'
    num_classes = len(config['classes'])
    class_names = config['classes']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger le modèle
    model = load_pretrained_model(model_path, num_classes).to(device)
    
    # Tester les images
    import glob
    test_images = glob.glob(f"{test_dir}/*.jpg") + glob.glob(f"{test_dir}/*.png")
    
    for img_path in test_images[:5]:  # Tester 5 images
        print(f"Testing: {img_path}")
        boxes, labels, scores, image = test_image(model, img_path, class_names, device)
        
        if len(boxes) > 0:
            print(f"Détections: {len(boxes)} objets")
            visualize_detections(image, boxes, labels, scores, class_names)
        else:
            print("Aucune détection")

if __name__ == "__main__":
    main()