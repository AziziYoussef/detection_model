import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import glob
import random
from tqdm import tqdm

# Vos classes
ORIGINAL_CLASSES = [
    'backpack', 'suitcase', 'handbag', 'cell phone', 'laptop',
    'book', 'umbrella', 'bottle', 'keyboard', 'remote'
]

EXTENDED_CLASSES = [
    'backpack', 'suitcase', 'handbag', 'tie', 'hair drier', 'toothbrush',
    'cell phone', 'laptop', 'keyboard', 'mouse', 'remote', 'tv',
    'bottle', 'cup', 'bowl', 'knife', 'spoon', 'fork', 'wine glass',
    'scissors', 'book', 'clock', 'umbrella', 'vase', 'potted plant',
    'bicycle', 'skateboard', 'sports ball'
]

def check_setup():
    """Vérifie que tout est en place"""
    
    print("🔍 Vérification de la configuration...")
    
    # Vérifier les annotations
    ann_file = 'coco_evaluation/annotations/instances_val2017.json'
    if not os.path.exists(ann_file):
        print("❌ Annotations manquantes!")
        return False
    
    # Vérifier les images
    img_dir = 'coco_evaluation/images/val2017'
    if not os.path.exists(img_dir):
        print("❌ Dossier images manquant!")
        return False
    
    images = glob.glob(f"{img_dir}/*.jpg")
    print(f"✅ Configuration OK: {len(images)} images disponibles")
    
    return len(images) > 0

def load_coco_and_select_images(max_images=50):
    """Charge COCO et sélectionne des images pertinentes"""
    
    print("📊 Chargement des annotations COCO...")
    ann_file = 'coco_evaluation/annotations/instances_val2017.json'
    coco = COCO(ann_file)
    
    # Vos classes disponibles
    available_classes = []
    class_ids = []
    
    for class_name in EXTENDED_CLASSES:
        cat_ids = coco.getCatIds(catNms=[class_name])
        if cat_ids:
            available_classes.append(class_name)
            class_ids.extend(cat_ids)
    
    print(f"✅ {len(available_classes)} classes trouvées dans COCO")
    
    # Trouver des images contenant ces classes
    selected_img_ids = set()
    
    for class_id in class_ids:
        img_ids = coco.getImgIds(catIds=[class_id])
        # Prendre un échantillon
        sample_size = min(5, len(img_ids))
        selected_for_class = random.sample(img_ids, sample_size)
        selected_img_ids.update(selected_for_class)
    
    # Limiter le total
    selected_img_ids = list(selected_img_ids)
    if len(selected_img_ids) > max_images:
        selected_img_ids = random.sample(selected_img_ids, max_images)
    
    # Vérifier que les images existent physiquement
    img_dir = 'coco_evaluation/images/val2017'
    available_images = []
    
    for img_id in selected_img_ids:
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if os.path.exists(img_path):
            available_images.append({
                'id': img_id,
                'path': img_path,
                'info': img_info
            })
    
    print(f"📷 {len(available_images)} images sélectionnées pour l'évaluation")
    
    return coco, available_images, available_classes

def load_model(model_path, num_classes):
    """Charge un modèle"""
    
    if not os.path.exists(model_path):
        return None
    
    model = detection_models.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Erreur chargement {os.path.basename(model_path)}: {e}")
        return None

def calculate_iou(box1, box2):
    """Calcule l'IoU entre deux boîtes [x1, y1, x2, y2]"""
    
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

def evaluate_model_on_image(model, image_path, coco, img_id, model_classes, device):
    """Évalue un modèle sur une image"""
    
    # Charger et prétraiter l'image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédiction
    model.to(device)
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Extraire les prédictions
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    
    # Filtrer par score
    conf_thresh = 0.3
    mask = pred_scores > conf_thresh
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]
    
    # Calculer la confiance moyenne
    avg_confidence = np.mean(pred_scores) if len(pred_scores) > 0 else 0.0
    
    # Obtenir les ground truth
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    annotations = coco.loadAnns(ann_ids)
    
    gt_boxes = []
    gt_classes = []
    
    # Créer le mapping des classes
    class_mapping = {}
    for i, class_name in enumerate(model_classes):
        cat_ids = coco.getCatIds(catNms=[class_name])
        if cat_ids:
            class_mapping[cat_ids[0]] = i + 1  # +1 car 0 est le fond
    
    for ann in annotations:
        if ann['category_id'] in class_mapping:
            bbox = ann['bbox']  # [x, y, width, height]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            gt_boxes.append([x1, y1, x2, y2])
            gt_classes.append(class_mapping[ann['category_id']])
    
    # Calculer les métriques
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    
    if num_gt == 0 and num_pred == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0, 'avg_conf': avg_confidence}
    
    if num_gt == 0:
        return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0, 'tp': 0, 'fp': num_pred, 'fn': 0, 'avg_conf': avg_confidence}
    
    if num_pred == 0:
        return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': num_gt, 'avg_conf': avg_confidence}
    
    # Calculer les matches avec IoU > 0.5
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
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'num_pred': num_pred,
        'num_gt': num_gt,
        'avg_conf': avg_confidence  # ← CORRECTION: Ajout du champ manquant
    }

def run_complete_evaluation():
    """Lance l'évaluation complète"""
    
    print("="*80)
    print("🏆 ÉVALUATION QUANTITATIVE COMPLÈTE SUR COCO")
    print("="*80)
    
    # Vérifier la configuration
    if not check_setup():
        return
    
    # Charger COCO et sélectionner les images
    coco, test_images, available_classes = load_coco_and_select_images(max_images=30)
    
    if not test_images:
        print("❌ Aucune image disponible pour l'évaluation")
        return
    
    # Modèles à évaluer
    models_to_test = [
        {
            'name': 'Original (10 classes)',
            'path': 'output_fast/fast_model_epoch_10.pth',
            'classes': ORIGINAL_CLASSES
        },
        {
            'name': 'Étendu - Meilleur',
            'path': 'output_extended_30/best_extended_model.pth',
            'classes': EXTENDED_CLASSES
        },
        {
            'name': 'Étendu - Époque 4',
            'path': 'output_extended_30/extended_model_epoch_4.pth',
            'classes': EXTENDED_CLASSES
        },
        {
            'name': 'Étendu - Final',
            'path': 'output_extended_30/extended_model_epoch_20.pth',
            'classes': EXTENDED_CLASSES
        }
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    
    # Résultats globaux
    all_results = {}
    
    for model_info in models_to_test:
        print(f"\n{'='*60}")
        print(f"🧪 ÉVALUATION: {model_info['name']}")
        print(f"{'='*60}")
        
        # Charger le modèle
        model = load_model(model_info['path'], len(model_info['classes']))
        
        if model is None:
            print(f"❌ Modèle non disponible: {model_info['name']}")
            continue
        
        # Évaluer sur toutes les images
        image_results = []
        
        for img_data in tqdm(test_images, desc="Évaluation"):
            try:
                result = evaluate_model_on_image(
                    model, img_data['path'], coco, img_data['id'],
                    model_info['classes'], device
                )
                image_results.append(result)
                
            except Exception as e:
                print(f"❌ Erreur sur {img_data['info']['file_name']}: {e}")
                continue
        
        if image_results:
            # Calculer les moyennes
            avg_precision = np.mean([r['precision'] for r in image_results])
            avg_recall = np.mean([r['recall'] for r in image_results])
            avg_f1 = np.mean([r['f1'] for r in image_results])
            avg_conf = np.mean([r['avg_conf'] for r in image_results])
            
            total_tp = sum([r['tp'] for r in image_results])
            total_fp = sum([r['fp'] for r in image_results])
            total_fn = sum([r['fn'] for r in image_results])
            
            # Sauvegarder les résultats
            all_results[model_info['name']] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'confidence': avg_conf,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'num_images': len(image_results)
            }
            
            print(f"📊 RÉSULTATS {model_info['name']}:")
            print(f"  • Précision: {avg_precision:.3f}")
            print(f"  • Rappel: {avg_recall:.3f}")
            print(f"  • F1-Score: {avg_f1:.3f}")
            print(f"  • Confiance: {avg_conf:.3f}")
            print(f"  • TP/FP/FN: {total_tp}/{total_fp}/{total_fn}")
        
        # Libérer la mémoire
        del model
        torch.cuda.empty_cache()
    
    # Comparaison finale
    if all_results:
        print(f"\n{'='*80}")
        print("🏆 COMPARAISON FINALE - MÉTRIQUES COCO")
        print(f"{'='*80}")
        print(f"{'Modèle':<25} {'Précision':<10} {'Rappel':<10} {'F1-Score':<10} {'Confiance':<10}")
        print("-" * 80)
        
        # Trier par F1-Score
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        for model_name, results in sorted_results:
            print(f"{model_name:<25} {results['precision']:<10.3f} "
                  f"{results['recall']:<10.3f} {results['f1']:<10.3f} "
                  f"{results['confidence']:<10.3f}")
        
        # Champion
        best_model = sorted_results[0]
        print(f"\n🏆 CHAMPION: {best_model[0]}")
        print(f"   📊 F1-Score: {best_model[1]['f1']:.3f}")
        print(f"   🎯 Précision: {best_model[1]['precision']:.3f}")
        print(f"   🔍 Rappel: {best_model[1]['recall']:.3f}")
        
        # Sauvegarder les résultats
        os.makedirs('coco_evaluation/results', exist_ok=True)
        with open('coco_evaluation/results/evaluation_results.json', 'w') as f:
            results_for_json = {}
            for model_name, results in all_results.items():
                results_for_json[model_name] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else int(v) if isinstance(v, (np.int32, np.int64)) else v for k, v in results.items()}

            json.dump(results_for_json, f, indent=2)
        
        print(f"\n💾 Résultats détaillés sauvegardés:")
        print(f"   📄 coco_evaluation/results/evaluation_results.json")
        
        # Analyse des forces/faiblesses
        print(f"\n📈 ANALYSE:")
        for model_name, results in sorted_results:
            if results['precision'] > results['recall']:
                print(f"  • {model_name}: Bon en précision, peut détecter plus d'objets")
            elif results['recall'] > results['precision']:
                print(f"  • {model_name}: Bon en détection, peut réduire les faux positifs")
            else:
                print(f"  • {model_name}: Équilibré entre précision et détection")

def main():
    print("="*80)
    print("🎯 ÉVALUATION QUANTITATIVE SUR COCO - VERSION CORRIGÉE")
    print("="*80)
    
    if not check_setup():
        print("\n❌ Configuration incomplète!")
        return
    
    print("\n✅ Configuration validée!")
    choice = input("Lancer l'évaluation complète? (y/n): ").lower().strip()
    
    if choice == 'y':
        run_complete_evaluation()
    else:
        print("Évaluation annulée.")

if __name__ == "__main__":
    main()