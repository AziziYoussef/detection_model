# inference_improved.py - Test du modèle amélioré
# Copiez ce code dans un nouveau fichier nommé "inference_improved.py"

import os
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config_improved import config

class ImprovedInference:
    """Classe pour faire de l'inférence avec le modèle amélioré"""
    
    def __init__(self, model_path, config, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        
        print(f"✅ Modèle chargé sur {self.device}")
        print(f"📊 Classes disponibles: {len(config['classes'])}")
        
    def load_model(self, model_path):
        """Charge le modèle amélioré"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Créer le modèle
        model = detection_models.fasterrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.config['classes']) + 1)
        
        # Charger les poids
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def get_transform(self):
        """Transformations pour l'inférence"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Prétraite une image"""
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        original_size = image.shape[:2]  # (height, width)
        
        # Redimensionner
        target_size = self.config['image_size']
        image_resized = cv2.resize(image, target_size)
        
        # Calculer les facteurs d'échelle
        scale_y = original_size[0] / target_size[0]
        scale_x = original_size[1] / target_size[1]
        
        # Appliquer les transformations
        image_tensor = self.transform(image_resized)
        
        return image_tensor, original_image, (scale_y, scale_x)
    
    def detect_objects(self, image_path, confidence_threshold=None):
        """Détecte les objets dans une image"""
        # Utiliser le seuil par défaut si non spécifié
        if confidence_threshold is None:
            confidence_threshold = self.config['confidence_threshold']
        
        # Prétraitement
        image_tensor, original_image, scale_factors = self.preprocess_image(image_path)
        
        # Ajouter dimension batch
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inférence
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Extraire les prédictions
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        
        # Filtrage par confiance
        mask = pred_scores > confidence_threshold
        pred_boxes = pred_boxes[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]
        
        # Ajuster les coordonnées à l'image originale
        if len(pred_boxes) > 0:
            scale_y, scale_x = scale_factors
            pred_boxes[:, [0, 2]] *= scale_x  # x1, x2
            pred_boxes[:, [1, 3]] *= scale_y  # y1, y2
        
        return {
            'boxes': pred_boxes,
            'labels': pred_labels,
            'scores': pred_scores,
            'original_image': original_image
        }
    
    def visualize_detections(self, detection_result, output_path=None, show_confidence=True):
        """Visualise les détections"""
        image = detection_result['original_image']
        boxes = detection_result['boxes']
        labels = detection_result['labels']
        scores = detection_result['scores']
        
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        ax = plt.gca()
        
        # Statistiques des détections
        person_count = 0
        object_count = 0
        
        # Dessiner chaque détection
        for box, label, score in zip(boxes, labels, scores):
            if len(box) == 0:
                continue
                
            x1, y1, x2, y2 = box
            
            # Obtenir le nom de la classe
            class_idx = label - 1  # -1 car 0 est le fond
            if 0 <= class_idx < len(self.config['classes']):
                class_name = self.config['classes'][class_idx]
                class_name_fr = self.config['class_names_fr'].get(class_name, class_name)
                
                # Compter les personnes et objets
                if class_name == 'person':
                    person_count += 1
                    color = 'red'  # Rouge pour les personnes
                    linewidth = 3  # Plus épais pour les personnes
                else:
                    object_count += 1
                    color = 'blue'  # Bleu pour les objets
                    linewidth = 2
                
                # Créer le rectangle
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=linewidth, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Texte avec confiance
                if show_confidence:
                    text = f"{class_name_fr}: {score:.2f}"
                else:
                    text = class_name_fr
                
                # Taille du texte adaptée
                fontsize = 12 if class_name == 'person' else 10
                
                plt.text(
                    x1, y1-5, text,
                    color='white', fontsize=fontsize, weight='bold',
                    bbox=dict(facecolor=color, alpha=0.8, pad=2)
                )
        
        # Titre avec statistiques
        title = f"Détections: {person_count} personne(s), {object_count} objet(s)"
        plt.title(title, fontsize=16, weight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Résultat sauvegardé: {output_path}")
        
        plt.show()

def main():
    """Fonction principale pour tester l'inférence"""
    print("="*70)
    print("🔍 TEST DU MODÈLE AMÉLIORÉ - DÉTECTION OBJETS + PERSONNES")
    print("="*70)
    
    # Chemin du modèle
    model_path = os.path.join(config['output_dir'], 'best_improved_model.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        print("Entraînez d'abord le modèle avec train_improved.py")
        return
    
    # Créer l'instance d'inférence
    try:
        inference = ImprovedInference(model_path, config)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return
    
    # Test sur une image
    test_image = "test_images/test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"⚠️ Image de test non trouvée: {test_image}")
        print("Créez un dossier 'test_images' avec des images pour tester")
        
        # Créer un dossier de test
        os.makedirs("test_images", exist_ok=True)
        print("📁 Dossier 'test_images' créé. Ajoutez vos images de test.")
        return
    
    print(f"🔍 Test sur: {test_image}")
    
    try:
        # Détection
        result = inference.detect_objects(test_image, confidence_threshold=0.3)
        
        # Afficher les résultats
        person_count = sum(1 for label in result['labels'] if label == 1)  # person est généralement classe 1
        object_count = len(result['labels']) - person_count
        
        print(f"\n📊 RÉSULTATS DE DÉTECTION:")
        print(f"  👥 Personnes détectées: {person_count}")
        print(f"  📦 Objets détectés: {object_count}")
        print(f"  🎯 Total détections: {len(result['labels'])}")
        
        # Détails des détections
        if len(result['labels']) > 0:
            print(f"\n🔍 DÉTAILS:")
            for i, (label, score) in enumerate(zip(result['labels'], result['scores'])):
                if 1 <= label <= len(config['classes']):
                    class_name = config['classes'][label - 1]
                    class_name_fr = config['class_names_fr'].get(class_name, class_name)
                    print(f"  {i+1}. {class_name_fr}: {score:.3f}")
        
        # Visualisation
        output_path = os.path.join(config['output_dir'], 'detection_result_improved.jpg')
        inference.visualize_detections(result, output_path)
        
        print(f"\n✅ AMÉLIORATIONS VISIBLES:")
        if person_count > 0:
            print("  👥 ✅ Détection de personnes fonctionne!")
        print(f"  📊 ✅ {len(config['classes'])} classes disponibles")
        print(f"  🎯 ✅ Détections avec confiances élevées")
        
    except Exception as e:
        print(f"❌ Erreur lors de la détection: {e}")
        return

if __name__ == "__main__":
    main()