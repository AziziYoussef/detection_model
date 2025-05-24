#!/usr/bin/env python3
"""
Script pour tester tous les modèles du dossier output_stable_training sur COCO
Usage: python test_stable_models_coco.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm
import time

# Configuration des modèles stable training
STABLE_MODELS_DIR = "output_stable_training"
COCO_EVAL_DIR = "coco_evaluation"

# Classes des modèles stable (28 classes)
STABLE_CLASSES = [
    'person', 'backpack', 'suitcase', 'handbag', 'tie',
    'umbrella', 'hair drier', 'toothbrush', 'cell phone',
    'laptop', 'keyboard', 'mouse', 'remote', 'tv',
    'clock', 'microwave', 'bottle', 'cup', 'bowl',
    'knife', 'spoon', 'fork', 'wine glass', 'refrigerator',
    'scissors', 'book', 'vase', 'chair'
]

def check_coco_setup():
    """Vérifie que COCO evaluation est configuré"""
    ann_file = os.path.join(COCO_EVAL_DIR, 'annotations', 'instances_val2017.json')
    img_dir = os.path.join(COCO_EVAL_DIR, 'images', 'val2017')
    
    if not os.path.exists(ann_file):
        print(f"❌ Annotations manquantes: {ann_file}")
        return False
    
    if not os.path.exists(img_dir):
        print(f"❌ Images manquantes: {img_dir}")
        return False
    
    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"✅ COCO configuré: {len(images)} images disponibles")
    return True

def find_stable_models():
    """Trouve tous les modèles dans output_stable_training"""
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
        
        # Taille du fichier
        size_mb = os.path.getsize(model_file) / (1024 * 1024)
        
        models.append({
            'name': filename,
            'path': model_file,
            'type': model_type,
            'priority': priority,
            'size_mb': size_mb,
            'display_name': f"{model_type} ({filename})"
        })
    
    # Trier par priorité puis par nom
    models.sort(key=lambda x: (x['priority'], x['name']))
    
    print(f"🔍 {len(models)} modèles trouvés dans {STABLE_MODELS_DIR}:")
    for i, model in enumerate(models, 1):
        print(f"  {i:2d}. {model['display_name']:<50} ({model['size_mb']:.1f} MB)")
    
    return models

def extract_epoch_number(filename):
    """Extrait le numéro d'époque du nom de fichier"""
    import re
    match = re.search(r'epoch_(\d+)', filename)
    return int(match.group(1)) if match else 0

def load_coco_test_data(max_images=30):
    """Charge les données de test COCO"""
    ann_file = os.path.join(COCO_EVAL_DIR, 'annotations', 'instances_val2017.json')
    img_dir = os.path.join(COCO_EVAL_DIR, 'images', 'val2017')
    
    print("📊 Chargement COCO...")
    coco = COCO(ann_file)
    
    # Trouver images avec nos classes
    available_class_ids = []
    for class_name in STABLE_CLASSES:
        cat_ids = coco.getCatIds(catNms=[class_name])
        if cat_ids:
            available_class_ids.extend(cat_ids)
    
    print(f"✅ {len(available_class_ids)} classes trouvées dans COCO")
    
    # Sélectionner images
    selected_img_ids = set()
    for class_id in available_class_ids:
        img_ids = coco.getImgIds(catIds=[class_id])
        sample_size = min(3, len(img_ids))  # 3 images par classe max
        import random
        random.seed(42)
        selected_for_class = random.sample(img_ids, sample_size)
        selected_img_ids.update(selected_for_class)
    
    # Limiter le total
    selected_img_ids = list(selected_img_ids)
    if len(selected_img_ids) > max_images:
        selected_img_ids = random.sample(selected_img_ids, max_images)
    
    # Vérifier existence physique
    test_images = []
    for img_id in selected_img_ids:
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if os.path.exists(img_path):
            test_images.append({
                'id': img_id,
                'path': img_path,
                'info': img_info
            })
    
    print(f"📷 {len(test_images)} images sélectionnées pour test")
    return coco, test_images

def load_stable_model(model_path, num_classes):
    """Charge un modèle stable"""
    try:
        model = detection_models.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        print(f"❌ Erreur chargement {os.path.basename(model_path)}: {e}")
        return None, None

def calculate_iou(box1, box2):
    """Calcule l'IoU entre deux boîtes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def evaluate_model_on_image(model, image_path, coco, img_id, device, confidence_threshold=0.3):
    """Évalue un modèle sur une image"""
    
    # Prétraitement
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédiction avec mesure de temps
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image_tensor)
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
    
    # Ground truth
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    annotations = coco.loadAnns(ann_ids)
    
    gt_boxes = []
    gt_classes = []
    
    # Mapping des classes
    class_mapping = {}
    for i, class_name in enumerate(STABLE_CLASSES):
        cat_ids = coco.getCatIds(catNms=[class_name])
        if cat_ids:
            class_mapping[cat_ids[0]] = i + 1
    
    for ann in annotations:
        if ann['category_id'] in class_mapping:
            bbox = ann['bbox']  # [x, y, width, height]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            gt_boxes.append([x1, y1, x2, y2])
            gt_classes.append(class_mapping[ann['category_id']])
    
    # Calcul des métriques
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    
    if num_gt == 0 and num_pred == 0:
        return {
            'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 
            'tp': 0, 'fp': 0, 'fn': 0, 
            'inference_time': inference_time,
            'avg_confidence': 1.0
        }
    
    if num_gt == 0:
        return {
            'precision': 0.0, 'recall': 1.0, 'f1': 0.0,
            'tp': 0, 'fp': num_pred, 'fn': 0,
            'inference_time': inference_time,
            'avg_confidence': float(np.mean(pred_scores)) if len(pred_scores) > 0 else 0.0
        }
    
    if num_pred == 0:
        return {
            'precision': 1.0, 'recall': 0.0, 'f1': 0.0,
            'tp': 0, 'fp': 0, 'fn': num_gt,
            'inference_time': inference_time,
            'avg_confidence': 0.0
        }
    
    # Matching avec IoU > 0.5
    tp = 0
    matched_gt = set()
    
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_iou > 0.5:
            tp += 1
            matched_gt.add(best_gt_idx)
    
    fp = num_pred - tp
    fn = num_gt - tp
    
    precision = tp / num_pred if num_pred > 0 else 0
    recall = tp / num_gt if num_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'inference_time': inference_time,
        'avg_confidence': float(np.mean(pred_scores)) if len(pred_scores) > 0 else 0.0
    }

def run_evaluation():
    """Lance l'évaluation complète"""
    print("="*80)
    print("🧪 TEST MODÈLES STABLE TRAINING SUR COCO")
    print("="*80)
    
    # Vérifications
    if not check_coco_setup():
        return
    
    models = find_stable_models()
    if not models:
        return
    
    # Sélection des modèles
    print(f"\n🎯 OPTIONS:")
    print(f"1. Tester TOUS les modèles ({len(models)})")
    print(f"2. Tester seulement le meilleur")
    print(f"3. Tester les 5 dernières époques")
    
    choice = input("Votre choix (1/2/3): ").strip()
    
    if choice == "2":
        selected_models = [m for m in models if 'best' in m['name']]
        if not selected_models:
            selected_models = models[:1]  # Premier modèle si pas de "best"
    elif choice == "3":
        epoch_models = [m for m in models if 'epoch' in m['name']]
        selected_models = epoch_models[-5:] if len(epoch_models) >= 5 else epoch_models
    else:
        selected_models = models
    
    print(f"\n✅ {len(selected_models)} modèles sélectionnés")
    
    # Charger données de test
    coco, test_images = load_coco_test_data()
    
    if not test_images:
        print("❌ Aucune image de test disponible")
        return
    
    print(f"\n🚀 Démarrage des tests...")
    print(f"📷 Images de test: {len(test_images)}")
    
    # Tests
    all_results = {}
    
    for i, model_info in enumerate(selected_models, 1):
        print(f"\n🧪 Test {i}/{len(selected_models)}: {model_info['display_name']}")
        
        # Charger modèle
        model, device = load_stable_model(model_info['path'], len(STABLE_CLASSES))
        
        if model is None:
            continue
        
        # Évaluer sur toutes les images
        image_results = []
        
        for img_data in tqdm(test_images, desc="Évaluation"):
            try:
                result = evaluate_model_on_image(
                    model, img_data['path'], coco, img_data['id'], device
                )
                image_results.append(result)
            except Exception as e:
                print(f"❌ Erreur sur {img_data['info']['file_name']}: {e}")
                continue
        
        if image_results:
            # Calculer moyennes
            avg_precision = float(np.mean([r['precision'] for r in image_results]))
            avg_recall = float(np.mean([r['recall'] for r in image_results]))
            avg_f1 = float(np.mean([r['f1'] for r in image_results]))
            avg_conf = float(np.mean([r['avg_confidence'] for r in image_results]))
            avg_time = float(np.mean([r['inference_time'] for r in image_results]))
            
            total_tp = int(sum([r['tp'] for r in image_results]))
            total_fp = int(sum([r['fp'] for r in image_results]))
            total_fn = int(sum([r['fn'] for r in image_results]))
            
            all_results[model_info['display_name']] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'confidence': avg_conf,
                'inference_time': avg_time,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'num_images': len(image_results),
                'file_size_mb': model_info['size_mb']
            }
            
            print(f"✅ F1: {avg_f1:.3f} | Précision: {avg_precision:.3f} | Rappel: {avg_recall:.3f}")
        
        # Libérer mémoire
        del model
        torch.cuda.empty_cache()
    
    # Affichage des résultats
    display_results(all_results)
    
    # Sauvegarde
    save_results(all_results)

def display_results(results):
    """Affiche les résultats de comparaison"""
    if not results:
        print("❌ Aucun résultat à afficher")
        return
    
    print(f"\n{'='*100}")
    print("🏆 RÉSULTATS COMPLETS - MODÈLES STABLE TRAINING")
    print(f"{'='*100}")
    
    # En-tête du tableau
    header = f"{'Modèle':<35} {'F1':<8} {'Précision':<10} {'Rappel':<8} {'Confiance':<10} {'Temps(ms)':<10} {'Taille(MB)':<12}"
    print(header)
    print("-" * 100)
    
    # Trier par F1-Score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for model_name, result in sorted_results:
        row = (f"{model_name:<35} "
               f"{result['f1']:<8.3f} "
               f"{result['precision']:<10.3f} "
               f"{result['recall']:<8.3f} "
               f"{result['confidence']:<10.3f} "
               f"{result['inference_time']:<10.0f} "
               f"{result['file_size_mb']:<12.1f}")
        print(row)
    
    # Champion
    if sorted_results:
        champion = sorted_results[0]
        print(f"\n🏆 CHAMPION: {champion[0]}")
        print(f"   📊 F1-Score: {champion[1]['f1']:.3f}")
        print(f"   🎯 Précision: {champion[1]['precision']:.3f}")
        print(f"   🔍 Rappel: {champion[1]['recall']:.3f}")
        print(f"   ⚡ Vitesse: {champion[1]['inference_time']:.0f}ms par image")
    
    # Analyse
    print(f"\n📈 ANALYSE:")
    best_f1 = max(results.values(), key=lambda x: x['f1'])
    best_speed = min(results.values(), key=lambda x: x['inference_time'])
    best_precision = max(results.values(), key=lambda x: x['precision'])
    
    print(f"   🎯 Meilleur F1-Score: {best_f1['f1']:.3f}")
    print(f"   ⚡ Plus rapide: {best_speed['inference_time']:.0f}ms")
    print(f"   🔍 Meilleure précision: {best_precision['precision']:.3f}")

def save_results(results):
    """Sauvegarde les résultats"""
    output_file = os.path.join(STABLE_MODELS_DIR, 'evaluation_results_coco.json')
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {output_file}")
        
        # Créer aussi un graphique
        create_comparison_chart(results)
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")

def create_comparison_chart(results):
    """Crée un graphique de comparaison"""
    if len(results) < 2:
        return
    
    model_names = [name.split('(')[0].strip() for name in results.keys()]  # Noms courts
    f1_scores = [result['f1'] for result in results.values()]
    precisions = [result['precision'] for result in results.values()]
    recalls = [result['recall'] for result in results.values()]
    times = [result['inference_time'] for result in results.values()]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # F1-Scores
    bars1 = ax1.bar(model_names, f1_scores, color='skyblue', alpha=0.8)
    ax1.set_title('F1-Score par Modèle', fontweight='bold')
    ax1.set_ylabel('F1-Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Ajouter valeurs sur les barres
    for bar, score in zip(bars1, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Précision vs Rappel
    ax2.scatter(precisions, recalls, s=100, alpha=0.7, c=f1_scores, cmap='viridis')
    ax2.set_xlabel('Précision')
    ax2.set_ylabel('Rappel')
    ax2.set_title('Précision vs Rappel', fontweight='bold')
    
    # Ajouter labels
    for i, name in enumerate(model_names):
        ax2.annotate(name.split()[1] if len(name.split()) > 1 else name[:10], 
                    (precisions[i], recalls[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    # Temps d'inférence
    bars3 = ax3.bar(model_names, times, color='lightcoral', alpha=0.8)
    ax3.set_title('Temps d\'Inférence par Modèle', fontweight='bold')
    ax3.set_ylabel('Temps (ms)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Performance globale (F1 / temps)
    efficiency = [f1/time*1000 for f1, time in zip(f1_scores, times)]  # F1 per second
    bars4 = ax4.bar(model_names, efficiency, color='lightgreen', alpha=0.8)
    ax4.set_title('Efficacité (F1-Score / Temps)', fontweight='bold')
    ax4.set_ylabel('Efficacité')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    chart_path = os.path.join(STABLE_MODELS_DIR, 'models_comparison_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Graphique sauvegardé: {chart_path}")

def main():
    """Fonction principale"""
    print("🧪 TEST MODÈLES STABLE TRAINING")
    print("="*50)
    
    if not os.path.exists(STABLE_MODELS_DIR):
        print(f"❌ Dossier non trouvé: {STABLE_MODELS_DIR}")
        return
    
    print(f"📁 Dossier modèles: {STABLE_MODELS_DIR}")
    print(f"📊 Classes testées: {len(STABLE_CLASSES)}")
    
    choice = input("\nLancer l'évaluation COCO des modèles stable? (y/n): ").lower().strip()
    
    if choice == 'y':
        run_evaluation()
    else:
        print("❌ Test annulé")

if __name__ == "__main__":
    main()
