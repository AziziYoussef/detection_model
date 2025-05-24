#!/usr/bin/env python3
"""
Test COMPLET du modèle Époque 30 sur TOUT le dataset val2017
Usage: python test_epoch30_full_evaluation.py
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Configuration
EPOCH_30_MODEL = "output_stable_training/stable_model_epoch_30.pth"
COCO_VAL_DIR = "coco_evaluation"
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

def check_setup():
    """Vérifie la configuration complète"""
    print("🔍 Vérification de la configuration...")
    
    # Vérifier le modèle epoch 30
    if not os.path.exists(EPOCH_30_MODEL):
        print(f"❌ Modèle epoch 30 manquant: {EPOCH_30_MODEL}")
        return False
    
    # Vérifier les annotations val2017
    ann_file = os.path.join(COCO_VAL_DIR, 'annotations', 'instances_val2017.json')
    if not os.path.exists(ann_file):
        print(f"❌ Annotations manquantes: {ann_file}")
        return False
    
    # Vérifier les images val2017
    img_dir = os.path.join(COCO_VAL_DIR, 'images', 'val2017')
    if not os.path.exists(img_dir):
        print(f"❌ Images manquantes: {img_dir}")
        return False
    
    # Compter les images
    import glob
    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    
    print(f"✅ Configuration complète vérifiée:")
    print(f"   🤖 Modèle: {os.path.basename(EPOCH_30_MODEL)}")
    print(f"   📊 Annotations: instances_val2017.json")
    print(f"   📷 Images: {len(images)} images val2017")
    
    return True

def load_epoch30_model():
    """Charge le modèle epoch 30"""
    print("🤖 Chargement du modèle Époque 30...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = detection_models.fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(STABLE_CLASSES) + 1)
        
        model.load_state_dict(torch.load(EPOCH_30_MODEL, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"✅ Modèle epoch 30 chargé sur {device}")
        return model, device
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        return None, None

def load_val2017_data():
    """Charge toutes les données val2017"""
    print("📊 Chargement des données val2017...")
    
    ann_file = os.path.join(COCO_VAL_DIR, 'annotations', 'instances_val2017.json')
    img_dir = os.path.join(COCO_VAL_DIR, 'images', 'val2017')
    
    # Charger COCO
    coco = COCO(ann_file)
    
    # Trouver nos classes dans COCO
    available_class_ids = []
    missing_classes = []
    
    for class_name in STABLE_CLASSES:
        cat_ids = coco.getCatIds(catNms=[class_name])
        if cat_ids:
            available_class_ids.append(cat_ids[0])
        else:
            missing_classes.append(class_name)
    
    print(f"✅ Classes trouvées: {len(available_class_ids)}/{len(STABLE_CLASSES)}")
    if missing_classes:
        print(f"⚠️ Classes manquantes: {missing_classes}")
    
    # Trouver TOUTES les images contenant nos classes
    all_img_ids = set()
    for class_id in available_class_ids:
        img_ids = coco.getImgIds(catIds=[class_id])
        all_img_ids.update(img_ids)
    
    all_img_ids = sorted(list(all_img_ids))
    
    # Vérifier existence physique des images
    valid_images = []
    for img_id in all_img_ids:
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if os.path.exists(img_path):
            valid_images.append({
                'id': img_id,
                'path': img_path,
                'info': img_info
            })
    
    print(f"📷 Images valides: {len(valid_images)} sur {len(all_img_ids)}")
    
    return coco, valid_images, available_class_ids

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

def evaluate_on_image(model, image_path, coco, img_id, class_ids, device, confidence_threshold=0.3):
    """Évalue le modèle sur une image"""
    
    # Prétraitement
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Redimensionner
        image_resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image_resized).unsqueeze(0).to(device)
        
    except Exception as e:
        print(f"❌ Erreur prétraitement {image_path}: {e}")
        return None
    
    # Prédiction
    start_time = time.time()
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
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
    
    # Remettre à l'échelle originale
    if len(pred_boxes) > 0:
        scale_x = original_size[0] / IMAGE_SIZE[0]
        scale_y = original_size[1] / IMAGE_SIZE[1]
        pred_boxes[:, [0, 2]] *= scale_x
        pred_boxes[:, [1, 3]] *= scale_y
    
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

def run_full_evaluation():
    """Lance l'évaluation complète"""
    print("="*80)
    print("🏆 ÉVALUATION COMPLÈTE - MODÈLE ÉPOQUE 30 SUR TOUT VAL2017")
    print("="*80)
    
    # Vérifications
    if not check_setup():
        return
    
    # Charger le modèle
    model, device = load_epoch30_model()
    if model is None:
        return
    
    # Charger les données
    coco, valid_images, class_ids = load_val2017_data()
    if not valid_images:
        print("❌ Aucune image valide trouvée")
        return
    
    print(f"\n🚀 DÉMARRAGE DE L'ÉVALUATION COMPLÈTE")
    print(f"📊 Total images à traiter: {len(valid_images)}")
    print(f"🤖 Modèle: Époque 30")
    print(f"🎯 Classes: {len(STABLE_CLASSES)}")
    
    # Seuil de confiance
    conf_thresh = input(f"\nSeuil de confiance (défaut 0.3): ").strip()
    try:
        confidence_threshold = float(conf_thresh) if conf_thresh else 0.3
    except ValueError:
        confidence_threshold = 0.3
    
    print(f"🎚️ Seuil de confiance: {confidence_threshold}")
    
    confirmation = input(f"\n⚠️ Cela va traiter {len(valid_images)} images (peut prendre du temps). Continuer? (y/n): ")
    if confirmation.lower() != 'y':
        print("❌ Évaluation annulée")
        return
    
    # Évaluation complète
    print(f"\n⏰ Début de l'évaluation... (estimé: {len(valid_images)*0.12/60:.1f} minutes)")
    
    all_results = []
    failed_images = 0
    total_inference_time = 0
    
    # Barre de progression
    pbar = tqdm(valid_images, desc="Évaluation Époque 30")
    
    for img_data in pbar:
        try:
            result = evaluate_on_image(
                model, img_data['path'], coco, img_data['id'], 
                class_ids, device, confidence_threshold
            )
            
            if result is not None:
                all_results.append(result)
                total_inference_time += result['inference_time']
                
                # Mise à jour barre de progression
                if len(all_results) % 100 == 0:
                    current_f1 = np.mean([r['f1'] for r in all_results])
                    pbar.set_postfix({
                        'F1': f"{current_f1:.3f}",
                        'Images': len(all_results),
                        'Échecs': failed_images
                    })
            else:
                failed_images += 1
                
        except Exception as e:
            failed_images += 1
            continue
    
    pbar.close()
    
    # Calcul des résultats finaux
    if not all_results:
        print("❌ Aucun résultat valide obtenu")
        return
    
    print(f"\n📊 CALCUL DES RÉSULTATS FINAUX...")
    
    # Moyennes globales
    avg_precision = float(np.mean([r['precision'] for r in all_results]))
    avg_recall = float(np.mean([r['recall'] for r in all_results]))
    avg_f1 = float(np.mean([r['f1'] for r in all_results]))
    avg_confidence = float(np.mean([r['avg_confidence'] for r in all_results]))
    avg_inference_time = float(total_inference_time / len(all_results))
    
    # Totaux
    total_tp = int(sum([r['tp'] for r in all_results]))
    total_fp = int(sum([r['fp'] for r in all_results]))
    total_fn = int(sum([r['fn'] for r in all_results]))
    
    # Affichage des résultats
    display_final_results(
        avg_precision, avg_recall, avg_f1, avg_confidence, avg_inference_time,
        total_tp, total_fp, total_fn, len(all_results), failed_images,
        confidence_threshold
    )
    
    # Sauvegarde
    save_complete_results(all_results, {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'confidence': avg_confidence,
        'inference_time': avg_inference_time,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'total_images': len(all_results),
        'failed_images': failed_images,
        'confidence_threshold': confidence_threshold
    })

def display_final_results(precision, recall, f1, confidence, inference_time, 
                         tp, fp, fn, total_images, failed_images, threshold):
    """Affiche les résultats finaux"""
    print(f"\n{'='*80}")
    print("🏆 RÉSULTATS FINAUX - ÉPOQUE 30 SUR VAL2017 COMPLET")
    print(f"{'='*80}")
    
    print(f"📊 MÉTRIQUES PRINCIPALES:")
    print(f"   🎯 Précision:     {precision:.4f} ({precision*100:.2f}%)")
    print(f"   🔍 Rappel:        {recall:.4f} ({recall*100:.2f}%)")
    print(f"   ⚖️ F1-Score:      {f1:.4f} ({f1*100:.2f}%)")
    print(f"   💪 Confiance:     {confidence:.4f} ({confidence*100:.2f}%)")
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"   ⏱️ Temps moyen:    {inference_time:.2f}ms par image")
    print(f"   🖥️ Vitesse:       {1000/inference_time:.1f} images/seconde")
    
    print(f"\n📈 DÉTAILS:")
    print(f"   ✅ Vrais Positifs:  {tp}")
    print(f"   ❌ Faux Positifs:   {fp}")
    print(f"   ❌ Faux Négatifs:   {fn}")
    print(f"   📷 Images traitées: {total_images}")
    print(f"   💥 Échecs:          {failed_images}")
    print(f"   🎚️ Seuil:          {threshold}")
    
    print(f"\n🎯 ÉVALUATION CONTEXTUELLE:")
    
    if f1 >= 0.50:
        evaluation = "🏆 EXCELLENT - Prêt pour production"
    elif f1 >= 0.40:
        evaluation = "✅ BON - Performance satisfaisante"
    elif f1 >= 0.30:
        evaluation = "⚠️ CORRECT - Améliorations possibles"
    else:
        evaluation = "❌ FAIBLE - Nécessite optimisation"
    
    print(f"   {evaluation}")
    print(f"   📊 F1-Score de {f1:.1%} sur {total_images} images")
    
    # Comparaison avec standards
    print(f"\n📚 COMPARAISON AVEC STANDARDS:")
    standards = {
        "Débutant": 0.35,
        "Bon": 0.55, 
        "Très bon": 0.72,
        "Excellent": 0.85
    }
    
    for level, score in standards.items():
        status = "✅" if f1 >= score else "❌"
        print(f"   {status} {level:<12} (F1 ≥ {score:.2f})")

def save_complete_results(all_results, summary):
    """Sauvegarde les résultats complets"""
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder résultats détaillés
    detailed_file = os.path.join(output_dir, "epoch30_full_evaluation_detailed.json")
    with open(detailed_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Sauvegarder résumé
    summary_file = os.path.join(output_dir, "epoch30_full_evaluation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Créer graphique de performance
    create_performance_chart(all_results, summary)
    
    print(f"\n💾 RÉSULTATS SAUVEGARDÉS:")
    print(f"   📄 Détaillé: {detailed_file}")
    print(f"   📋 Résumé:   {summary_file}")
    print(f"   📊 Graphique: {output_dir}/epoch30_performance_chart.png")

def create_performance_chart(all_results, summary):
    """Crée un graphique de performance"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Distribution F1-Score
    f1_scores = [r['f1'] for r in all_results]
    ax1.hist(f1_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(summary['f1'], color='red', linestyle='--', linewidth=2, label=f'Moyenne: {summary["f1"]:.3f}')
    ax1.set_xlabel('F1-Score')
    ax1.set_ylabel('Nombre d\'images')
    ax1.set_title('Distribution des F1-Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Précision vs Rappel
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    ax2.scatter(precisions, recalls, alpha=0.5, s=10)
    ax2.set_xlabel('Précision')
    ax2.set_ylabel('Rappel')
    ax2.set_title('Précision vs Rappel')
    ax2.grid(True, alpha=0.3)
    
    # Temps d'inférence
    inference_times = [r['inference_time'] for r in all_results]
    ax3.hist(inference_times, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(summary['inference_time'], color='red', linestyle='--', linewidth=2, 
                label=f'Moyenne: {summary["inference_time"]:.1f}ms')
    ax3.set_xlabel('Temps d\'inférence (ms)')
    ax3.set_ylabel('Nombre d\'images')
    ax3.set_title('Distribution des Temps d\'Inférence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Résumé des métriques
    metrics = ['Précision', 'Rappel', 'F1-Score', 'Confiance']
    values = [summary['precision'], summary['recall'], summary['f1'], summary['confidence']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
    ax4.set_ylabel('Score')
    ax4.set_title('Métriques Finales - Époque 30')
    ax4.set_ylim(0, 1)
    
    # Ajouter valeurs sur les barres
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/epoch30_performance_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Fonction principale"""
    print("🏆 ÉVALUATION COMPLÈTE - ÉPOQUE 30")
    print("="*50)
    
    print("Ce script va évaluer le modèle Époque 30 sur TOUT le dataset val2017")
    print("⚠️ Cela peut prendre du temps (plusieurs minutes à heures selon votre GPU)")
    
    choice = input("\nLancer l'évaluation complète? (y/n): ").lower().strip()
    
    if choice == 'y':
        run_full_evaluation()
    else:
        print("❌ Évaluation annulée")

if __name__ == "__main__":
    main()
