# compare_models.py - Comparaison simple entre ancien et nouveau modèle
# Copiez ce code dans un nouveau fichier nommé "compare_models.py"

import os
import torch
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Configurations
try:
    from config_extended import config as old_config
except ImportError:
    old_config = None
    print("⚠️ config_extended.py non trouvé, comparaison limitée")

from config_improved import config as new_config

class SimpleModelComparison:
    """Comparaison simple entre deux modèles"""
    
    def __init__(self, old_model_path, new_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🔄 Chargement des modèles...")
        self.old_model = self.load_model(old_model_path, old_config) if old_config else None
        self.new_model = self.load_model(new_model_path, new_config) if new_config else None
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✅ Modèles chargés!")
    
    def load_model(self, model_path, config):
        """Charge un modèle"""
        if not os.path.exists(model_path) or not config:
            print(f"⚠️ Modèle non trouvé: {model_path}")
            return None
        
        try:
            model = detection_models.fasterrcnn_resnet50_fpn(pretrained=False)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(config['classes']) + 1)
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Erreur chargement {model_path}: {e}")
            return None
    
    def preprocess_image(self, image_path, target_size):
        """Prétraite une image"""
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Redimensionner
        image = image.resize(target_size)
        image_tensor = self.transform(image)
        
        return image_tensor, original_image
    
    def detect_with_model(self, model, image_tensor, config, confidence_threshold=0.3):
        """Détection avec un modèle"""
        if model is None:
            return {'boxes': [], 'labels': [], 'scores': []}
        
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            predictions = model(image_tensor)
        
        # Filtrer par confiance
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        
        mask = pred_scores > confidence_threshold
        return {
            'boxes': pred_boxes[mask],
            'labels': pred_labels[mask],
            'scores': pred_scores[mask]
        }
    
    def compare_on_image(self, image_path, confidence_threshold=0.3):
        """Compare les deux modèles sur une image"""
        print(f"\n🔍 Comparaison sur: {os.path.basename(image_path)}")
        
        # Préparation des images
        if self.old_model and old_config:
            old_tensor, _ = self.preprocess_image(image_path, old_config['image_size'])
        else:
            old_tensor = None
        
        new_tensor, original_image = self.preprocess_image(image_path, new_config['image_size'])
        
        # Détections avec mesure du temps
        results = {}
        
        if self.old_model and old_config:
            start_time = time.time()
            old_result = self.detect_with_model(self.old_model, old_tensor, old_config, confidence_threshold)
            old_time = time.time() - start_time
            results['old'] = {'result': old_result, 'time': old_time, 'config': old_config}
        
        if self.new_model:
            start_time = time.time()
            new_result = self.detect_with_model(self.new_model, new_tensor, new_config, confidence_threshold)
            new_time = time.time() - start_time
            results['new'] = {'result': new_result, 'time': new_time, 'config': new_config}
        
        # Analyser et visualiser
        self.analyze_and_visualize(original_image, results, image_path)
        
        return results
    
    def analyze_and_visualize(self, original_image, results, image_path):
        """Analyse et visualise les résultats"""
        
        if 'old' in results and 'new' in results:
            # Comparaison complète
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Ancien modèle
            old_result = results['old']['result']
            old_time = results['old']['time']
            old_config = results['old']['config']
            
            ax1.imshow(original_image)
            ax1.set_title(f'Ancien Modèle\n{len(old_result["labels"])} détections | {old_time:.3f}s', 
                         fontsize=14, weight='bold')
            
            self.draw_detections(ax1, old_result, old_config, 'red')
            
            # Nouveau modèle
            new_result = results['new']['result']
            new_time = results['new']['time']
            new_config = results['new']['config']
            
            ax2.imshow(original_image)
            
            # Vérifier la détection de personnes
            person_detected = any(
                new_config['classes'][label-1] == 'person' 
                for label in new_result['labels'] 
                if 1 <= label <= len(new_config['classes'])
            )
            
            title = f'Nouveau Modèle Amélioré\n{len(new_result["labels"])} détections | {new_time:.3f}s'
            if person_detected:
                title += ' | 👥'
            ax2.set_title(title, fontsize=14, weight='bold')
            
            self.draw_detections(ax2, new_result, new_config, 'green', highlight_persons=True)
            
            ax1.axis('off')
            ax2.axis('off')
            
            # Résumé des améliorations
            improvement = len(new_result['labels']) - len(old_result['labels'])
            speed_change = (old_time - new_time) / old_time * 100 if old_time > 0 else 0
            
            improvement_text = f"""
AMÉLIORATIONS:
• Détections: {improvement:+d}
• Vitesse: {speed_change:+.1f}%
• Classes: {len(old_config['classes'])} → {len(new_config['classes'])}
• Personnes: {'✅' if person_detected else '❌'}
"""
            
            plt.figtext(0.5, 0.02, improvement_text, ha='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            
        else:
            # Affichage du nouveau modèle seulement
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            new_result = results['new']['result']
            new_time = results['new']['time']
            new_config = results['new']['config']
            
            ax.imshow(original_image)
            
            person_detected = any(
                new_config['classes'][label-1] == 'person' 
                for label in new_result['labels'] 
                if 1 <= label <= len(new_config['classes'])
            )
            
            title = f'Modèle Amélioré\n{len(new_result["labels"])} détections | {new_time:.3f}s'
            if person_detected:
                title += ' | 👥 Personne détectée!'
            ax.set_title(title, fontsize=14, weight='bold')
            
            self.draw_detections(ax, new_result, new_config, 'green', highlight_persons=True)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Sauvegarder
        output_path = f"comparison_result_{os.path.basename(image_path)}"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Comparaison sauvegardée: {output_path}")
        plt.show()
    
    def draw_detections(self, ax, result, config, color, highlight_persons=False):
        """Dessine les détections sur un axe"""
        for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
            if len(box) == 0 or not (1 <= label <= len(config['classes'])):
                continue
            
            x1, y1, x2, y2 = box
            class_name = config['classes'][label - 1]
            
            # Couleur spéciale pour les personnes
            if highlight_persons and class_name == 'person':
                detection_color = 'red'
                linewidth = 3
            else:
                detection_color = color
                linewidth = 2
            
            # Rectangle
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=linewidth, edgecolor=detection_color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Texte
            class_name_fr = config.get('class_names_fr', {}).get(class_name, class_name)
            ax.text(
                x1, y1-5, f"{class_name_fr}: {score:.2f}",
                color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor=detection_color, alpha=0.8, pad=1)
            )
    
    def print_summary(self, results):
        """Affiche un résumé des résultats"""
        print("\n📊 RÉSUMÉ DE COMPARAISON")
        print("="*50)
        
        if 'old' in results:
            old_result = results['old']['result']
            print(f"Ancien modèle: {len(old_result['labels'])} détections")
        
        if 'new' in results:
            new_result = results['new']['result']
            new_config = results['new']['config']
            
            person_count = sum(
                1 for label in new_result['labels']
                if 1 <= label <= len(new_config['classes']) and new_config['classes'][label-1] == 'person'
            )
            object_count = len(new_result['labels']) - person_count
            
            print(f"Nouveau modèle: {len(new_result['labels'])} détections")
            print(f"  • Personnes: {person_count}")
            print(f"  • Objets: {object_count}")
            print(f"  • Classes disponibles: {len(new_config['classes'])}")
            
            if person_count > 0:
                print("✅ Détection de personnes fonctionne!")

def main():
    """Fonction principale de comparaison"""
    print("="*70)
    print("⚔️ COMPARAISON ANCIEN VS NOUVEAU MODÈLE")
    print("="*70)
    
    # Chemins des modèles
    old_model_path = "output_extended_30/best_extended_model.pth"  # Ajustez selon votre modèle
    new_model_path = "output_improved_with_persons/best_improved_model.pth"
    
    # Créer l'instance de comparaison
    comparator = SimpleModelComparison(old_model_path, new_model_path)
    
    # Test sur une image
    test_image = "test_images/test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"⚠️ Image de test non trouvée: {test_image}")
        print("Créez un dossier 'test_images' avec des images")
        
        # Essayer avec d'autres images possibles
        possible_images = [
            "test_image.jpg",
            "example.jpg", 
            "sample.jpg"
        ]
        
        for img in possible_images:
            if os.path.exists(img):
                test_image = img
                break
        else:
            print("❌ Aucune image de test trouvée")
            return
    
    print(f"🔍 Test sur: {test_image}")
    
    try:
        # Comparaison
        results = comparator.compare_on_image(test_image)
        comparator.print_summary(results)
        
        print(f"\n✅ COMPARAISON TERMINÉE!")
        print(f"📊 Vérifiez l'image de comparaison générée")
        
    except Exception as e:
        print(f"❌ Erreur lors de la comparaison: {e}")

if __name__ == "__main__":
    main()