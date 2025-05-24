# compare_all_models.py - Test comparatif de tous vos modèles

#!/usr/bin/env python3
"""
Script pour tester une image sur TOUS les modèles STABLE uniquement
Usage: python test_stable_models_image.py
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

# Configuration modèles STABLE
STABLE_MODELS_DIR = "output_stable_training"
IMAGE_SIZE = (320, 320)

# Classes des modèles stable (28 classes)
STABLE_CLASSES = [
    'person', 'backpack', 'suitcase', 'handbag', 'tie',
    'umbrella', 'hair drier', 'toothbrush', 'cell phone',
    'laptop', 'keyboard', 'mouse', 'remote', 'tv',
    'clock', 'microwave', 'bottle', 'cup', 'bowl',
    'knife', 'spoon', 'fork', 'wine glass', 'refrigerator',
    'scissors', 'book', 'vase', 'chair'
]

# Noms français pour affichage
CLASSES_FR = {
    'person': 'Personne',
    'backpack': 'Sac à dos',
    'suitcase': 'Valise',
    'handbag': 'Sac à main',
    'tie': 'Cravate',
    'hair drier': 'Sèche-cheveux',
    'toothbrush': 'Brosse à dents',
    'cell phone': 'Téléphone',
    'laptop': 'Ordinateur portable',
    'keyboard': 'Clavier',
    'mouse': 'Souris',
    'remote': 'Télécommande',
    'tv': 'Télévision',
    'bottle': 'Bouteille',
    'cup': 'Tasse',
    'bowl': 'Bol',
    'knife': 'Couteau',
    'spoon': 'Cuillère',
    'fork': 'Fourchette',
    'wine glass': 'Verre',
    'scissors': 'Ciseaux',
    'book': 'Livre',
    'clock': 'Horloge',
    'umbrella': 'Parapluie',
    'vase': 'Vase',
    'chair': 'Chaise',
    'microwave': 'Micro-ondes',
    'refrigerator': 'Réfrigérateur'
}

def find_stable_models():
    """Trouve tous les modèles stable"""
    if not os.path.exists(STABLE_MODELS_DIR):
        print(f"❌ Dossier non trouvé: {STABLE_MODELS_DIR}")
        return []
    
    model_files = glob.glob(os.path.join(STABLE_MODELS_DIR, "*.pth"))
    
    models = []
    for model_file in model_files:
        filename = os.path.basename(model_file)
        
        # Extraire info du nom
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
    
    # Trier par priorité puis par époque
    models.sort(key=lambda x: (x['priority'], extract_epoch_number(x['name'])))
    
    print(f"🔍 {len(models)} modèles STABLE trouvés:")
    for i, model in enumerate(models, 1):
        print(f"  {i:2d}. {model['display_name']:<20} ({model['name']})")
    
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

def preprocess_image(image_path):
    """Prétraite une image"""
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"📷 Image: {os.path.basename(image_path)} - {image.size}")
    except Exception as e:
        print(f"❌ Erreur image: {e}")
        raise
    
    original_image = np.array(image)
    original_size = image.size
    
    # Redimensionner vers la taille d'entraînement
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
    
    inference_time = (time.time() - start_time) * 1000  # en ms
    
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
    person_count = 0
    object_count = 0
    class_counts = {}
    
    for label in pred_labels:
        if 1 <= label <= len(STABLE_CLASSES):
            class_name = STABLE_CLASSES[label - 1]
            if class_name == 'person':
                person_count += 1
            else:
                object_count += 1
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return {
        'boxes': pred_boxes,
        'labels': pred_labels,
        'scores': pred_scores,
        'inference_time': inference_time,
        'person_count': person_count,
        'object_count': object_count,
        'total_detections': len(pred_labels),
        'class_counts': class_counts,
        'avg_confidence': np.mean(pred_scores) if len(pred_scores) > 0 else 0.0
    }

def select_test_image():
    """Sélectionne une image de test"""
    # Dossiers possibles
    image_dirs = ['test_images', 'images', 'samples', '.']
    
    selected_dir = None
    for dir_name in image_dirs:
        if os.path.exists(dir_name):
            images = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                images.extend(glob.glob(os.path.join(dir_name, f'*{ext}')))
                images.extend(glob.glob(os.path.join(dir_name, f'*{ext.upper()}')))
            
            if images:
                selected_dir = dir_name
                break
    
    if not selected_dir:
        print("❌ Aucun dossier d'images trouvé")
        print("📁 Créez un dossier 'test_images' avec des images")
        return None
    
    # Lister les images disponibles
    images = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        images.extend(glob.glob(os.path.join(selected_dir, f'*{ext}')))
        images.extend(glob.glob(os.path.join(selected_dir, f'*{ext.upper()}')))
    
    print(f"\n📁 Images disponibles dans '{selected_dir}':")
    for i, img_path in enumerate(images, 1):
        img_name = os.path.basename(img_path)
        try:
            with Image.open(img_path) as img:
                size_info = f"{img.size[0]}x{img.size[1]}"
        except:
            size_info = "?"
        print(f"  {i:2d}. {img_name:<30} ({size_info})")
    
    # Sélection
    while True:
        try:
            choice = input(f"\nChoisissez une image (1-{len(images)}) ou 'q' pour quitter: ").strip()
            if choice.lower() == 'q':
                return None
            
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                return images[idx]
            else:
                print(f"❌ Choisissez un nombre entre 1 et {len(images)}")
        except ValueError:
            print("❌ Entrez un nombre valide")

def visualize_stable_comparison(original_image, results_dict, image_name, confidence_threshold):
    """Visualise la comparaison des modèles stable"""
    num_models = len(results_dict)
    
    if num_models == 0:
        print("❌ Aucun résultat à afficher")
        return
    
    # Calculer la grille optimale
    cols = min(4, num_models)  # Maximum 4 colonnes
    rows = (num_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # S'assurer que axes est toujours une liste
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
        axes = [ax for row in axes for ax in row]
    else:
        axes = axes.flatten()
    
    model_names = list(results_dict.keys())
    
    for i, (model_name, result) in enumerate(results_dict.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Afficher l'image
        ax.imshow(original_image)
        
        if result is not None:
            # Dessiner les détections
            draw_detections_on_axis(ax, result, model_name)
            
            # Titre avec statistiques
            title = f"{model_name}\n"
            title += f"👥 {result['person_count']} • 📦 {result['object_count']} • "
            title += f"⚡ {result['inference_time']:.0f}ms"
            ax.set_title(title, fontsize=10, weight='bold', pad=10)
        else:
            ax.set_title(f"{model_name}\n❌ Erreur", fontsize=10, color='red')
        
        ax.axis('off')
    
    # Masquer les axes non utilisés
    for i in range(len(results_dict), len(axes)):
        axes[i].axis('off')
    
    # Titre général
    fig.suptitle(f'Comparaison Modèles STABLE - {image_name}\nSeuil: {confidence_threshold}', 
                fontsize=16, weight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Sauvegarder
    output_name = f"stable_comparison_{os.path.splitext(image_name)[0]}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"💾 Comparaison sauvegardée: {output_name}")
    
    plt.show()

def draw_detections_on_axis(ax, result, model_name):
    """Dessine les détections sur un axe"""
    boxes = result['boxes']
    labels = result['labels']
    scores = result['scores']
    
    for box, label, score in zip(boxes, labels, scores):
        if len(box) == 0:
            continue
        
        x1, y1, x2, y2 = box
        
        if 1 <= label <= len(STABLE_CLASSES):
            class_name = STABLE_CLASSES[label - 1]
            
            # Couleur selon le type
            if class_name == 'person':
                color = 'red'
                linewidth = 2
            else:
                color = 'lime'
                linewidth = 1.5
            
            # Rectangle
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=linewidth, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Texte (seulement pour les détections de confiance élevée)
            if score > 0.4:
                class_name_fr = CLASSES_FR.get(class_name, class_name)
                ax.text(x1, y1-2, f"{class_name_fr}: {score:.2f}", 
                       color='white', fontsize=8, weight='bold',
                       bbox=dict(facecolor=color, alpha=0.8, pad=1))

def create_performance_table(results_dict):
    """Crée un tableau de performance"""
    print(f"\n{'='*80}")
    print("📊 PERFORMANCE DES MODÈLES STABLE")
    print(f"{'='*80}")
    
    # En-tête
    header = f"{'Modèle':<25} {'Personnes':<9} {'Objets':<7} {'Total':<6} {'Temps':<8} {'Conf Moy':<8}"
    print(header)
    print("-" * 80)
    
    # Trier par nombre total de détections (décroissant)
    sorted_results = sorted(results_dict.items(), 
                          key=lambda x: x[1]['total_detections'] if x[1] else 0, 
                          reverse=True)
    
    for model_name, result in sorted_results:
        if result is not None:
            row = (f"{model_name:<25} "
                  f"{result['person_count']:<9} "
                  f"{result['object_count']:<7} "
                  f"{result['total_detections']:<6} "
                  f"{result['inference_time']:<8.0f} "
                  f"{result['avg_confidence']:<8.3f}")
            print(row)
        else:
            print(f"{model_name:<25} {'❌ ERREUR':<40}")
    
    # Champion
    if sorted_results and sorted_results[0][1] is not None:
        champion = sorted_results[0]
        print(f"\n🏆 CHAMPION DÉTECTIONS: {champion[0]}")
        print(f"   📊 {champion[1]['total_detections']} détections")
        print(f"   ⚡ {champion[1]['inference_time']:.0f}ms")
        print(f"   🎯 Confiance moyenne: {champion[1]['avg_confidence']:.3f}")

def main():
    """Fonction principale"""
    print("="*80)
    print("🧪 TEST IMAGE - MODÈLES STABLE UNIQUEMENT")
    print("="*80)
    
    # Vérifier le dossier
    if not os.path.exists(STABLE_MODELS_DIR):
        print(f"❌ Dossier non trouvé: {STABLE_MODELS_DIR}")
        return
    
    # Trouver les modèles stable
    models = find_stable_models()
    if not models:
        return
    
    # Sélectionner l'image
    image_path = select_test_image()
    if not image_path:
        return
    
    # Seuil de confiance
    conf_thresh = input(f"\nSeuil de confiance (défaut 0.3): ").strip()
    try:
        confidence_threshold = float(conf_thresh) if conf_thresh else 0.3
    except ValueError:
        confidence_threshold = 0.3
    
    print(f"\n🚀 Démarrage du test...")
    print(f"📷 Image: {os.path.basename(image_path)}")
    print(f"🎯 Seuil: {confidence_threshold}")
    print(f"🤖 Modèles STABLE: {len(models)}")
    
    # Prétraiter l'image une seule fois
    image_tensor, original_image, scale_factors = preprocess_image(image_path)
    
    # Tester tous les modèles stable
    results = {}
    
    for i, model_info in enumerate(models, 1):
        model_name = model_info['display_name']
        print(f"\n🧪 Test {i}/{len(models)}: {model_name}")
        
        # Charger et tester le modèle
        model, device = load_stable_model(model_info['path'])
        result = detect_with_stable_model(model, image_tensor, scale_factors, device, confidence_threshold)
        
        results[model_name] = result
        
        if result:
            print(f"   ✅ {result['total_detections']} détections en {result['inference_time']:.0f}ms")
            if result['person_count'] > 0:
                print(f"   👥 {result['person_count']} personne(s) détectée(s)")
        else:
            print(f"   ❌ Échec")
        
        # Libérer la mémoire
        if model:
            del model
            torch.cuda.empty_cache()
    
    # Créer la visualisation comparative
    print(f"\n🎨 Création de la visualisation...")
    visualize_stable_comparison(original_image, results, os.path.basename(image_path), confidence_threshold)
    
    # Afficher le tableau de performance
    create_performance_table(results)
    
    print(f"\n✅ Test des modèles STABLE terminé!")

if __name__ == "__main__":
    main()