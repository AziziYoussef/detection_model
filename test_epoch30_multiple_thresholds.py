#!/usr/bin/env python3
"""
Test COMPLET du modèle Époque 30 avec SEUILS MULTIPLES (0.5 à 0.7)
Usage: python test_epoch30_thresholds_comparison.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

# SEUILS À TESTER (de 0.5 à 0.7)
CONFIDENCE_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7]

# Classes des modèles stable (28 classes)
STABLE_CLASSES = [
    'person', 'backpack', 'suitcase', 'handbag', 'tie',
    'umbrella', 'hair drier', 'toothbrush', 'cell phone',
    'laptop', 'keyboard', 'mouse', 'remote', 'tv',
    'clock', 'microwave', 'bottle', 'cup', 'bowl',
    'knife', 'spoon', 'fork', 'wine glass', 'refrigerator',
    'scissors', 'book', 'vase', 'chair'
]

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
    print(f"   🎯 Seuils à tester: {CONFIDENCE_THRESHOLDS}")
    
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

def evaluate_on_image_all_thresholds(model, image_path, coco, img_id, class_ids, device):
    """Évalue le modèle sur une image avec TOUS les seuils"""
    
    # Prétraitement (une seule fois)
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
        return None
    
    # Prédiction (une seule fois)
    start_time = time.time()
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    inference_time = (time.time() - start_time) * 1000
    
    # Extraire prédictions BRUTES
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    
    # Remettre à l'échelle originale
    if len(pred_boxes) > 0:
        scale_x = original_size[0] / IMAGE_SIZE[0]
        scale_y = original_size[1] / IMAGE_SIZE[1]
        pred_boxes[:, [0, 2]] *= scale_x
        pred_boxes[:, [1, 3]] *= scale_y
    
    # Ground truth (une seule fois)
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
    
    # ÉVALUER POUR CHAQUE SEUIL
    results_by_threshold = {}
    
    for threshold in CONFIDENCE_THRESHOLDS:
        # Filtrer par confiance pour ce seuil
        mask = pred_scores > threshold
        filtered_boxes = pred_boxes[mask]
        filtered_labels = pred_labels[mask]
        filtered_scores = pred_scores[mask]
        
        # Calcul des métriques pour ce seuil
        num_gt = len(gt_boxes)
        num_pred = len(filtered_boxes)
        
        if num_gt == 0 and num_pred == 0:
            result = {
                'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'tp': 0, 'fp': 0, 'fn': 0,
                'avg_confidence': 1.0
            }
        elif num_gt == 0:
            result = {
                'precision': 0.0, 'recall': 1.0, 'f1': 0.0,
                'tp': 0, 'fp': num_pred, 'fn': 0,
                'avg_confidence': float(np.mean(filtered_scores)) if len(filtered_scores) > 0 else 0.0
            }
        elif num_pred == 0:
            result = {
                'precision': 1.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'fp': 0, 'fn': num_gt,
                'avg_confidence': 0.0
            }
        else:
            # Matching avec IoU > 0.5
            tp = 0
            matched_gt = set()
            
            for pred_box in filtered_boxes:
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
            
            result = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'avg_confidence': float(np.mean(filtered_scores)) if len(filtered_scores) > 0 else 0.0
            }
        
        results_by_threshold[threshold] = result
    
    # Ajouter le temps d'inférence à tous les résultats
    for threshold_result in results_by_threshold.values():
        threshold_result['inference_time'] = inference_time
    
    return results_by_threshold

def run_multi_threshold_evaluation():
    """Lance l'évaluation avec plusieurs seuils"""
    print("="*80)
    print("🎯 ÉVALUATION MULTI-SEUILS - ÉPOQUE 30 (SEUILS 0.5 À 0.7)")
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
    
    print(f"\n🚀 DÉMARRAGE DE L'ÉVALUATION MULTI-SEUILS")
    print(f"📊 Total images à traiter: {len(valid_images)}")
    print(f"🎯 Seuils à tester: {CONFIDENCE_THRESHOLDS}")
    print(f"🤖 Modèle: Époque 30")
    
    confirmation = input(f"\n⚠️ Cela va traiter {len(valid_images)} images avec {len(CONFIDENCE_THRESHOLDS)} seuils. Continuer? (y/n): ")
    if confirmation.lower() != 'y':
        print("❌ Évaluation annulée")
        return
    
    # Évaluation complète
    print(f"\n⏰ Début de l'évaluation multi-seuils...")
    
    # Stockage des résultats par seuil
    results_by_threshold = {threshold: [] for threshold in CONFIDENCE_THRESHOLDS}
    failed_images = 0
    total_inference_time = 0
    
    # Barre de progression
    pbar = tqdm(valid_images, desc="Évaluation Multi-Seuils")
    
    for img_data in pbar:
        try:
            # Évaluer avec TOUS les seuils en une fois
            image_results = evaluate_on_image_all_thresholds(
                model, img_data['path'], coco, img_data['id'], 
                class_ids, device
            )
            
            if image_results is not None:
                # Ajouter les résultats à chaque seuil
                for threshold, result in image_results.items():
                    results_by_threshold[threshold].append(result)
                
                # Temps d'inférence (identique pour tous les seuils)
                total_inference_time += list(image_results.values())[0]['inference_time']
                
                # Mise à jour barre de progression
                if len(results_by_threshold[0.5]) % 100 == 0:
                    # Prendre F1 du seuil 0.5 pour affichage
                    current_f1 = np.mean([r['f1'] for r in results_by_threshold[0.5]])
                    pbar.set_postfix({
                        'F1@0.5': f"{current_f1:.3f}",
                        'Images': len(results_by_threshold[0.5]),
                        'Échecs': failed_images
                    })
            else:
                failed_images += 1
                
        except Exception as e:
            failed_images += 1
            continue
    
    pbar.close()
    
    # Calcul des résultats finaux pour chaque seuil
    if not results_by_threshold[0.5]:  # Vérifier avec le premier seuil
        print("❌ Aucun résultat valide obtenu")
        return
    
    print(f"\n📊 CALCUL DES RÉSULTATS FINAUX POUR CHAQUE SEUIL...")
    
    final_results = {}
    
    for threshold in CONFIDENCE_THRESHOLDS:
        results = results_by_threshold[threshold]
        
        if results:
            # Moyennes pour ce seuil
            avg_precision = float(np.mean([r['precision'] for r in results]))
            avg_recall = float(np.mean([r['recall'] for r in results]))
            avg_f1 = float(np.mean([r['f1'] for r in results]))
            avg_confidence = float(np.mean([r['avg_confidence'] for r in results]))
            
            # Totaux pour ce seuil
            total_tp = int(sum([r['tp'] for r in results]))
            total_fp = int(sum([r['fp'] for r in results]))
            total_fn = int(sum([r['fn'] for r in results]))
            
            final_results[threshold] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'confidence': avg_confidence,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'total_images': len(results),
                'failed_images': failed_images,
                'inference_time': total_inference_time / len(results)
            }
    
    # Affichage des résultats comparatifs
    display_threshold_comparison_results(final_results)
    
    # Sauvegarde
    save_threshold_comparison_results(final_results, results_by_threshold)

def display_threshold_comparison_results(final_results):
    """Affiche les résultats comparatifs pour tous les seuils"""
    print(f"\n{'='*100}")
    print("🎯 COMPARAISON DES SEUILS DE CONFIANCE - ÉPOQUE 30")
    print(f"{'='*100}")
    
    # En-tête du tableau
    header = f"{'Seuil':<8} {'Précision':<12} {'Rappel':<10} {'F1-Score':<10} {'Confiance':<12} {'TP':<6} {'FP':<6} {'FN':<6}"
    print(header)
    print("-" * 100)
    
    # Afficher chaque seuil
    best_f1_threshold = None
    best_f1_score = 0
    
    for threshold in CONFIDENCE_THRESHOLDS:
        if threshold in final_results:
            result = final_results[threshold]
            
            # Trouver le meilleur F1-Score
            if result['f1'] > best_f1_score:
                best_f1_score = result['f1']
                best_f1_threshold = threshold
            
            row = (f"{threshold:<8.2f} "
                   f"{result['precision']:<12.4f} "
                   f"{result['recall']:<10.4f} "
                   f"{result['f1']:<10.4f} "
                   f"{result['confidence']:<12.4f} "
                   f"{result['tp']:<6d} "
                   f"{result['fp']:<6d} "
                   f"{result['fn']:<6d}")
            
            # Mettre en évidence le meilleur
            if threshold == best_f1_threshold:
                print(f"🏆 {row}")
            else:
                print(f"   {row}")
    
    # Analyse détaillée du meilleur seuil
    if best_f1_threshold is not None:
        best_result = final_results[best_f1_threshold]
        
        print(f"\n🏆 MEILLEUR SEUIL IDENTIFIÉ: {best_f1_threshold}")
        print(f"{'='*50}")
        print(f"   🎯 F1-Score:      {best_result['f1']:.4f} ({best_result['f1']*100:.2f}%)")
        print(f"   📊 Précision:     {best_result['precision']:.4f} ({best_result['precision']*100:.2f}%)")
        print(f"   🔍 Rappel:        {best_result['recall']:.4f} ({best_result['recall']*100:.2f}%)")
        print(f"   💪 Confiance:     {best_result['confidence']:.4f} ({best_result['confidence']*100:.2f}%)")
        print(f"   ⚡ Vitesse:       {1000/best_result['inference_time']:.1f} images/seconde")
        
        print(f"\n📈 RECOMMANDATIONS:")
        print(f"   ✅ Utilisez le seuil {best_f1_threshold} pour la production")
        print(f"   🎯 Performance optimale: F1={best_result['f1']:.3f}")
        
        if best_result['precision'] > best_result['recall']:
            print(f"   📊 Profil: PRÉCISION (moins de fausses alarmes)")
        else:
            print(f"   🔍 Profil: RAPPEL (détection exhaustive)")
    
    # Analyse des tendances
    print(f"\n📊 ANALYSE DES TENDANCES:")
    precisions = [final_results[t]['precision'] for t in CONFIDENCE_THRESHOLDS if t in final_results]
    recalls = [final_results[t]['recall'] for t in CONFIDENCE_THRESHOLDS if t in final_results]
    f1s = [final_results[t]['f1'] for t in CONFIDENCE_THRESHOLDS if t in final_results]
    
    if len(precisions) > 1:
        if precisions[-1] > precisions[0]:
            print("   📈 Précision AUGMENTE avec le seuil (normal)")
        if recalls[-1] < recalls[0]:
            print("   📉 Rappel DIMINUE avec le seuil (normal)")
        
        max_f1_idx = f1s.index(max(f1s))
        optimal_threshold = CONFIDENCE_THRESHOLDS[max_f1_idx]
        print(f"   🎯 Équilibre optimal à {optimal_threshold}")

def save_threshold_comparison_results(final_results, all_results):
    """Sauvegarde les résultats de comparaison"""
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder résumé comparatif
    comparison_file = os.path.join(output_dir, "epoch30_threshold_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Sauvegarder résultats détaillés par seuil
    for threshold in CONFIDENCE_THRESHOLDS:
        if threshold in all_results:
            detailed_file = os.path.join(output_dir, f"epoch30_threshold_{threshold:.2f}_detailed.json")
            with open(detailed_file, 'w') as f:
                json.dump(all_results[threshold], f, indent=2)
    
    # Créer graphiques de comparaison
    create_threshold_comparison_charts(final_results)
    
    print(f"\n💾 RÉSULTATS SAUVEGARDÉS:")
    print(f"   📋 Comparaison: {comparison_file}")
    print(f"   📊 Graphiques: {output_dir}/threshold_comparison_*.png")

def create_threshold_comparison_charts(final_results):
    """Crée les graphiques de comparaison des seuils"""
    
    thresholds = sorted(final_results.keys())
    precisions = [final_results[t]['precision'] for t in thresholds]
    recalls = [final_results[t]['recall'] for t in thresholds]
    f1s = [final_results[t]['f1'] for t in thresholds]
    confidences = [final_results[t]['confidence'] for t in thresholds]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Évolution des métriques
    ax1.plot(thresholds, precisions, 'b-o', label='Précision', linewidth=2, markersize=8)
    ax1.plot(thresholds, recalls, 'r-s', label='Rappel', linewidth=2, markersize=8)
    ax1.plot(thresholds, f1s, 'g-^', label='F1-Score', linewidth=2, markersize=8)
    ax1.set_xlabel('Seuil de Confiance')
    ax1.set_ylabel('Score')
    ax1.set_title('Évolution des Métriques par Seuil')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Précision vs Rappel
    ax2.plot(recalls, precisions, 'ko-', linewidth=2, markersize=8)
    for i, threshold in enumerate(thresholds):
        ax2.annotate(f'{threshold}', (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Rappel')
    ax2.set_ylabel('Précision')
    ax2.set_title('Courbe Précision-Rappel')
    ax2.grid(True, alpha=0.3)
    
    # F1-Score par seuil (barres)
    bars = ax3.bar([str(t) for t in thresholds], f1s, 
                   color=['gold' if f == max(f1s) else 'skyblue' for f in f1s],
                   alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Seuil de Confiance')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score par Seuil')
    ax3.grid(True, alpha=0.3)
    
    # Ajouter valeurs sur les barres
    for bar, f1 in zip(bars, f1s):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Confiance moyenne par seuil
    ax4.plot(thresholds, confidences, 'purple', marker='D', linewidth=2, markersize=8)
    ax4.set_xlabel('Seuil de Confiance')
    ax4.set_ylabel('Confiance Moyenne')
    ax4.set_title('Confiance Moyenne par Seuil')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/threshold_comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Fonction principale"""
    print("🎯 ÉVALUATION MULTI-SEUILS - ÉPOQUE 30")
    print("="*60)
    
    print("Ce script va évaluer le modèle Époque 30 avec plusieurs seuils:")
    print(f"Seuils testés: {CONFIDENCE_THRESHOLDS}")
    print("Objectif: Trouver le seuil optimal pour votre cas d'usage")
    
    choice = input("\nLancer l'évaluation multi-seuils? (y/n): ").lower().strip()
    
    if choice == 'y':
        run_multi_threshold_evaluation()
    else:
        print("❌ Évaluation annulée")

if __name__ == "__main__":
    main()
