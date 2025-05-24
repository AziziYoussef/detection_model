#!/usr/bin/env python3
"""
Script pour test VISUEL sur images test2017 (sans métriques)
Usage: python test_stable_visual_test2017.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import json
import random

# Configuration
STABLE_MODELS_DIR = "output_stable_training"
COCO_TEST_DIR = "coco_evaluation"
IMAGE_SIZE = (320, 320)

# Classes des modèles stable
STABLE_CLASSES = [
    'person', 'backpack', 'suitcase', 'handbag', 'tie',
    'umbrella', 'hair drier', 'toothbrush', 'cell phone',
    'laptop', 'keyboard', 'mouse', 'remote', 'tv',
    'clock', 'microwave', 'bottle', 'cup', 'bowl',
    'knife', 'spoon', 'fork', 'wine glass', 'refrigerator',
    'scissors', 'book', 'vase', 'chair'
]

CLASSES_FR = {
    'person': 'Personne', 'backpack': 'Sac à dos', 'suitcase': 'Valise',
    'handbag': 'Sac à main', 'tie': 'Cravate', 'hair drier': 'Sèche-cheveux',
    'toothbrush': 'Brosse à dents', 'cell phone': 'Téléphone',
    'laptop': 'Ordinateur portable', 'keyboard': 'Clavier', 'mouse': 'Souris',
    'remote': 'Télécommande', 'tv': 'Télévision', 'bottle': 'Bouteille',
    'cup': 'Tasse', 'bowl': 'Bol', 'knife': 'Couteau', 'spoon': 'Cuillère',
    'fork': 'Fourchette', 'wine glass': 'Verre', 'scissors': 'Ciseaux',
    'book': 'Livre', 'clock': 'Horloge', 'umbrella': 'Parapluie',
    'vase': 'Vase', 'chair': 'Chaise', 'microwave': 'Micro-ondes',
    'refrigerator': 'Réfrigérateur'
}

def check_test2017_setup():
    """Vérifie la configuration test2017"""
    img_dir = os.path.join(COCO_TEST_DIR, 'images', 'test2017')
    info_file = os.path.join(COCO_TEST_DIR, 'annotations', 'image_info_test2017.json')
    
    if not os.path.exists(img_dir):
        print(f"❌ Images manquantes: {img_dir}")
        return False
    
    if not os.path.exists(info_file):
        print(f"❌ Infos manquantes: {info_file}")
        return False
    
    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"✅ Test2017 configuré: {len(images)} images disponibles")
    return True, len(images)

def load_test2017_info():
    """Charge les informations des images test2017"""
    info_file = os.path.join(COCO_TEST_DIR, 'annotations', 'image_info_test2017.json')
    
    with open(info_file, 'r') as f:
        data = json.load(f)
    
    return data['images']

def find_stable_models():
    """Trouve les modèles stable"""
    if not os.path.exists(STABLE_MODELS_DIR):
        return []
    
    model_files = glob.glob(os.path.join(STABLE_MODELS_DIR, "*.pth"))
    models = []
    
    for model_file in model_files:
        filename = os.path.basename(model_file)
        
        if 'best' in filename:
            model_type = "🏆 Meilleur"
            priority = 1
        elif 'epoch' in filename:
            epoch_num = extract_epoch_number(filename)
            model_type = f"📅 Époque {epoch_num}"
            priority = 2
        else:
            model_type = "📦 Standard"
            priority = 3
        
        models.append({
            'name': filename,
            'path': model_file,
            'type': model_type,
            'priority': priority,
            'display_name': f"{model_type}"
        })
    
    models.sort(key=lambda x: (x['priority'], extract_epoch_number(x['name'])))
    return models

def extract_epoch_number(filename):
    """Extrait le numéro d'époque"""
    import re
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0

def load_stable_model(model_path):
    """Charge un modèle stable"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = detection_models.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(STABLE_CLASSES) + 1)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        print(f"❌ Erreur chargement {os.path.basename(model_path)}: {e}")
        return None, None

def select_random_test_images(image_info_list, num_images=5):
    """Sélectionne des images aléatoires"""
    selected = random.sample(image_info_list, min(num_images, len(image_info_list)))
    
    print(f"🎲 {len(selected)} images test2017 sélectionnées aléatoirement:")
    for i, img_info in enumerate(selected, 1):
        print(f"  {i}. {img_info['file_name']} ({img_info['width']}x{img_info['height']})")
    
    return selected

def preprocess_image(image_path):
    """Prétraite une image"""
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    original_size = image.size
    
    # Redimensionner
    image_resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    
    # Facteurs d'échelle
    scale_x = original_size[0] / IMAGE_SIZE[0]
    scale_y = original_size[1] / IMAGE_SIZE[1]
    
    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_resized)
    return image_tensor, original_image, (scale_x, scale_y)

def detect_with_stable_model(model, image_tensor, scale_factors, device, confidence_threshold=0.3):
    """Détection avec un modèle stable"""
    if model is None:
        return None
    
    start_time = time.time()
    
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(device)
        predictions = model(image_batch)
    
    inference_time = (time.time() - start_time) * 1000
    
    # Extraire prédictions
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    
    # Filtrer par confiance
    mask = pred_scores > confidence_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]
    
    # Remettre à l'échelle
    if len(pred_boxes) > 0:
        scale_x, scale_y = scale_factors
        pred_boxes[:, [0, 2]] *= scale_x
        pred_boxes[:, [1, 3]] *= scale_y
    
    # Analyser les détections
    person_count = sum(1 for label in pred_labels if 1 <= label <= len(STABLE_CLASSES) and STABLE_CLASSES[label-1] == 'person')
    object_count = len(pred_labels) - person_count
    
    return {
        'boxes': pred_boxes,
        'labels': pred_labels,
        'scores': pred_scores,
        'inference_time': inference_time,
        'person_count': person_count,
        'object_count': object_count,
        'total_detections': len(pred_labels),
        'avg_confidence': np.mean(pred_scores) if len(pred_scores) > 0 else 0.0
    }

def visualize_test_results(original_image, results_dict, image_name, confidence_threshold):
    """Visualise les résultats sur image test"""
    num_models = len(results_dict)
    
    if num_models == 0:
        return
    
    # Grille
    cols = min(3, num_models)
    rows = (num_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
        axes = [ax for row in axes for ax in row]
    else:
        axes = axes.flatten()
    
    for i, (model_name, result) in enumerate(results_dict.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        ax.imshow(original_image)
        
        if result is not None:
            # Dessiner détections
            for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
                if len(box) == 0 or not (1 <= label <= len(STABLE_CLASSES)):
                    continue
                
                x1, y1, x2, y2 = box
                class_name = STABLE_CLASSES[label - 1]
                
                color = 'red' if class_name == 'person' else 'lime'
                linewidth = 2 if class_name == 'person' else 1.5
                
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=linewidth, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                if score > 0.4:
                    class_name_fr = CLASSES_FR.get(class_name, class_name)
                    ax.text(x1, y1-2, f"{class_name_fr}: {score:.2f}", 
                           color='white', fontsize=8, weight='bold',
                           bbox=dict(facecolor=color, alpha=0.8, pad=1))
            
            # Titre
            title = f"{model_name}\n👥 {result['person_count']} • 📦 {result['object_count']} • ⚡ {result['inference_time']:.0f}ms"
            ax.set_title(title, fontsize=10, weight='bold')
        else:
            ax.set_title(f"{model_name}\n❌ Erreur", fontsize=10, color='red')
        
        ax.axis('off')
    
    # Masquer axes non utilisés
    for i in range(len(results_dict), len(axes)):
        axes[i].axis('off')
    
    # Titre général
    fig.suptitle(f'Test Visuel sur COCO Test2017 - {image_name}\nSeuil: {confidence_threshold}', 
                fontsize=16, weight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Sauvegarder
    output_name = f"./test2017_output/test2017_visual_{os.path.splitext(image_name)[0]}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"💾 Résultat sauvegardé: {output_name}")
    
    plt.show()

def main():
    """Fonction principale"""
    print("="*80)
    print("🎨 TEST VISUEL - MODÈLES STABLE SUR COCO TEST2017")
    print("="*80)
    print("⚠️  Note: Test2017 n'a pas d'annotations -> Test visuel uniquement")
    
    # Vérification
    setup_ok, num_images = check_test2017_setup()
    if not setup_ok:
        return
    
    # Trouver modèles
    models = find_stable_models()
    if not models:
        print("❌ Aucun modèle stable trouvé!")
        return
    
    print(f"\n🔍 {len(models)} modèles STABLE trouvés:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['display_name']}")
    
    # Charger infos images
    print(f"\n📊 Chargement des informations test2017...")
    image_info_list = load_test2017_info()
    print(f"✅ {len(image_info_list)} images disponibles")
    
    # Sélectionner images aléatoires
    num_test_images = min(10, len(image_info_list))
    selected_images = select_random_test_images(image_info_list, num_test_images)
    
    # Seuil de confiance
    conf_thresh = input(f"\nSeuil de confiance (défaut 0.3): ").strip()
    try:
        confidence_threshold = float(conf_thresh) if conf_thresh else 0.3
    except ValueError:
        confidence_threshold = 0.3
    
    print(f"\n🚀 Démarrage du test visuel...")
    print(f"🎯 Seuil: {confidence_threshold}")
    
    # Tester chaque image
    for img_idx, img_info in enumerate(selected_images, 1):
        print(f"\n{'='*60}")
        print(f"📷 IMAGE {img_idx}/{len(selected_images)}: {img_info['file_name']}")
        print(f"{'='*60}")
        
        # Chemin image
        img_path = os.path.join(COCO_TEST_DIR, 'images', 'test2017', img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"❌ Image non trouvée: {img_path}")
            continue
        
        # Prétraiter
        image_tensor, original_image, scale_factors = preprocess_image(img_path)
        
        # Tester tous les modèles sur cette image
        results = {}
        
        for model_info in models:
            model_name = model_info['display_name']
            print(f"🧪 Test: {model_name}... ", end='')
            
            model, device = load_stable_model(model_info['path'])
            result = detect_with_stable_model(model, image_tensor, scale_factors, device, confidence_threshold)
            
            results[model_name] = result
            
            if result:
                print(f"✅ {result['total_detections']} détections ({result['inference_time']:.0f}ms)")
            else:
                print("❌")
            
            if model:
                del model
                torch.cuda.empty_cache()
        
        # Visualiser
        visualize_test_results(original_image, results, img_info['file_name'], confidence_threshold)
    
    print(f"\n✅ Test visuel terminé sur {len(selected_images)} images!")
    print("📊 Résultats: Détections visuelles uniquement (pas de métriques de précision)")

if __name__ == "__main__":
    main()